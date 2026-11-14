#!/usr/bin/env python3
"""dev-c4-human: human-like city following closed-loop sim.

usage: human_city_sim.py TIP_ROOT BUILD_DIR OUT_JSON CANDIDATE_JSON [personalities] [scenario-substrings]

CANDIDATE_JSON: {"name": ..., "human": {HumanCityConfig fields} | null, "release": bool}
  human null -> no human-city module (dev-c4 9c6e321c06 behavior)
  release    -> Toyota brake-hold release in the carcontroller (LAUNCH CHANGE)
Adapted from dev-c4-sng-test sng_city_sim.py (d6045f6359).

Runs this tree's real LongitudinalMpc (host acados build), planner
pieces (caps, cruise accel, positive-accel rise limit,
HumanCity) wired the same way longitudinal_planner.py wires them, real
LongControl, and the Toyota TSS2 carcontroller long path emulated
line-for-line (4 m/s³ rate limit, TSS2 PID, a_ego_future). Radar lead:
aLeadK is a 0.3 s lag of the true accel and aLeadTau follows radard
(1.5 when |a| < 0.5, else filtered toward 0 with 0.45 s). Plant: 0.05 s
delay + 0.25 s lag, hybrid actuator delay 0.05 s. The model's e2e
candidate is not simulated (always-Chill build). Host-only; not
road-tested.
"""
import importlib.util, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace

TIP = Path(sys.argv[1]).resolve()
BUILD = Path(sys.argv[2])
OUT = Path(sys.argv[3])
CAND = json.loads(sys.argv[4]) if sys.argv[4].lstrip().startswith("{") else json.loads(Path(sys.argv[4]).read_text())
MPH = 0.44704
FT = 0.3048

sys.path[:0] = [str(TIP), str(TIP / "opendbc_repo"), str(TIP / "msgq_repo"), "/tmp/test2-pydeps"]
helper = TIP / "openpilot/sunnypilot/selfdrive/controls/tests/test3_c3x_equiv_sim.py"
old = sys.argv[:]
sys.argv = [old[0], str(TIP), str(BUILD)]
spec = importlib.util.spec_from_file_location("c3x_helper", helper)
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
sys.argv = old
H._setup(BUILD, TIP)

import numpy as np
from openpilot.cereal import log
from opendbc.car.structs import car
import openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc as LM
import openpilot.selfdrive.controls.lib.longitudinal_planner as LP
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from opendbc.car.common.pid import PIDController
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car import rate_limit
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_accel_curve import smooth_accel_cap

try:
  from opendbc.sunnypilot.car.toyota.launch_release import release_prev_accel
except Exception:
  release_prev_accel = None
from openpilot.sunnypilot.selfdrive.controls.lib import devc4_human_city as HC

P = log.LongitudinalPersonality
PNAME = {P.aggressive: "aggressive", P.standard: "standard", P.relaxed: "relaxed"}
LongCtrlState = car.CarControl.Actuators.LongControlState
CTRL_T = ModelConstants.T_IDXS[:CONTROL_N]
ACCEL_MIN, ACCEL_MAX = -3.5, 2.0


HUMAN_PLAN_N = int(np.searchsorted(LM.T_IDXS, 2.0)) + 1


def stop_gap(p):
  return LM.STOP_DISTANCE + LM.get_lead_stop_extra(p, False)


def eq_gap(p, v):
  return float(LM.get_T_FOLLOW(p, False, v) * v + LM.STOP_DISTANCE)


def follow_gap(v_ego, v_lead, t):
  return float(LM.get_safe_obstacle_distance(v_ego, t) - LM.get_stopped_equivalence_factor(max(v_lead, 0.0)))


# Scenario: ego v0, lead v0, gap0 (None -> stock equilibrium / set stop gap),
# v_cruise, lead segments, optional cut-in, optional radar presence windows.
#   ("acc", a, v_target) | ("hold", s) | ("dist", a_up, a_dn, D)
#   lost=[(t0, t1)]: lead not reported (present False) in [t0, t1)
#   appear=t: lead not reported before t
def S(name, segs, ev0=0.0, lv0=0.0, gap0=None, v_cruise=45 * MPH, cut=None, t_tail=10.0, noise=False,
      lost=None, appear=None, kind="city", headway_mph=None, set_gap_check=True):
  return dict(name=name, segs=segs, ev0=ev0, lv0=lv0, gap0=gap0, v_cruise=v_cruise, cut=cut, t_tail=t_tail,
              noise=noise, lost=lost or [], appear=appear, kind=kind, headway_mph=headway_mph,
              set_gap_check=set_gap_check)


def accordion(peaks, ups, dns, wait=4.0, hold=2.0):
  segs = [("hold", 2.0)]
  for vp, au, ad in zip(peaks, ups, dns):
    segs += [("acc", au, vp * MPH), ("hold", hold), ("acc", ad, 0.0), ("hold", wait)]
  return segs


def signal_cycle(v_peak, n=3, up=1.5, dn=-1.8, green=12.0, red=20.0):
  segs = [("hold", 3.0)]
  for _ in range(n):
    segs += [("acc", up, v_peak * MPH), ("hold", green), ("acc", dn, 0.0), ("hold", red)]
  return segs


ROLLSTOP_PAUSES = (0.5, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 1.0, 2.0)


def rollstop(dists, pauses=ROLLSTOP_PAUSES, up=1.0, dn=-1.5):
  segs = [("hold", 2.0)]
  for i, d in enumerate(dists):
    segs += [("dist", up, dn, float(d)), ("hold", pauses[i % len(pauses)])]
  segs += [("hold", 4.0)]
  return segs


SCENARIOS = [
  # steady following: time headway (gap / v) after settling
  *[S(f"follow_{m}mph", [("hold", 45.0)], ev0=m * MPH, lv0=m * MPH, kind="headway", headway_mph=m, t_tail=0.0,
      v_cruise=(m + 5) * MPH)
    for m in (15, 25, 35, 45)],
  # signal queues on a 40-45 mph arterial (lights every ~1/4-1/2 mile)
  S("signal_queue_40mph", signal_cycle(40, up=1.5, dn=-1.8)),
  S("signal_queue_35mph_firm", signal_cycle(35, up=2.0, dn=-2.5)),
  S("approach_stopped_queue_40mph", [("hold", 1.0)], ev0=40 * MPH, lv0=0.0, gap0=120.0, t_tail=25.0),
  S("approach_stopped_queue_45mph", [("hold", 1.0)], ev0=45 * MPH, lv0=0.0, gap0=140.0, t_tail=25.0),
  S("lead_slows_to_stop_40mph", [("hold", 4.0), ("acc", -1.5, 0.0), ("hold", 8.0)], ev0=40 * MPH, lv0=40 * MPH),
  S("lead_hard_brake_40mph_3.5", [("hold", 4.0), ("acc", -3.5, 0.0), ("hold", 8.0)], ev0=40 * MPH, lv0=40 * MPH),
  # beyond openpilot's -3.5 m/s² ACCEL_MIN: needs the driver / AEB, reported only
  S("lead_hard_brake_40mph_4.5", [("hold", 4.0), ("acc", -4.5, 0.0), ("hold", 8.0)], ev0=40 * MPH, lv0=40 * MPH,
    kind="beyond_limit"),
  S("lead_hard_brake_25mph", [("hold", 4.0), ("acc", -4.0, 0.0), ("hold", 8.0)], ev0=25 * MPH, lv0=25 * MPH),
  # stop-and-go
  S("accordion_0_15mph", accordion([8, 15, 12, 5, 15], [1.0, 1.5, 1.2, 0.8, 2.0], [-1.5, -2.5, -2.0, -1.5, -3.0])),
  S("accordion_noisy_radar", accordion([10, 15, 6, 12], [1.2, 1.5, 0.8, 1.0], [-2.0, -2.5, -1.5, -3.0]), noise=True),
  S("rolling_5_25mph", [("hold", 3.0)] + sum([[("acc", 1.0, 25 * MPH), ("hold", 2.0), ("acc", -1.5, 5 * MPH), ("hold", 1.0)]
                                              for _ in range(4)], []), ev0=5 * MPH, lv0=5 * MPH),
  S("launch_to35mph", [("hold", 2.0), ("acc", 1.5, 35 * MPH), ("hold", 10.0)]),
  S("stopped_lead_hold", [("hold", 15.0)]),
  # cut-ins
  S("cutin_30mph_10m", [("hold", 12.0)], ev0=30 * MPH, lv0=30 * MPH, cut=dict(t=3.0, gap=10.0, v=28 * MPH),
    set_gap_check=False),
  S("cutin_12mph_then_stop", [("hold", 4.5), ("acc", -1.5, 0.0), ("hold", 6.0)], ev0=12 * MPH, lv0=12 * MPH,
    cut=dict(t=3.0, gap=7.0, v=10 * MPH), set_gap_check=False),
  # --- Nov 13 near-miss family (test3 d79496cac9: lead crept and stopped, ego kept going)
  S("nm_creep_1.5m_stop", [("hold", 2.0), ("dist", 1.0, -1.5, 1.5), ("hold", 6.0)], kind="nearmiss"),
  S("nm_creep_3m_stop", [("hold", 2.0), ("dist", 1.0, -1.5, 3.0), ("hold", 6.0)], kind="nearmiss"),
  S("nm_creep_3m_slow_up", [("hold", 2.0), ("dist", 0.5, -1.5, 3.0), ("hold", 6.0)], kind="nearmiss"),
  S("nm_creep_then_hard_brake", [("hold", 2.0), ("dist", 1.0, -3.5, 3.0), ("hold", 6.0)], kind="nearmiss"),
  S("nm_start_then_stop_4mph", [("hold", 2.0), ("acc", 1.0, 4 * MPH), ("acc", -4.0, 0.0), ("hold", 6.0)], kind="nearmiss"),
  S("nm_start_then_stop_8mph", [("hold", 2.0), ("acc", 1.5, 8 * MPH), ("acc", -4.0, 0.0), ("hold", 6.0)], kind="nearmiss"),
  S("nm_lead_lost_during_creep", [("hold", 2.0), ("dist", 1.0, -2.0, 3.0), ("hold", 6.0)], lost=[(3.0, 4.0)],
    kind="nearmiss"),
  S("nm_lead_lost_while_stopping_15mph", [("hold", 3.0), ("acc", -3.0, 0.0), ("hold", 8.0)], ev0=15 * MPH,
    lv0=15 * MPH, lost=[(3.2, 3.8)], kind="nearmiss"),
  S("nm_lead_appears_late_stopped_30mph", [("hold", 1.0)], ev0=30 * MPH, lv0=0.0, gap0=55.0, appear=0.0, t_tail=10.0,
    kind="nearmiss"),
  S("nm_lead_appears_late_crawl_10mph", [("hold", 1.0), ("acc", -1.0, 0.0), ("hold", 6.0)], ev0=10 * MPH,
    lv0=3 * MPH, gap0=16.0, appear=0.0, kind="nearmiss"),
  S("nm_cutin_crawl_close", [("hold", 3.0), ("acc", -2.0, 0.0), ("hold", 6.0)], ev0=5 * MPH, lv0=5 * MPH,
    cut=dict(t=2.0, gap=4.0, v=4 * MPH), kind="nearmiss", set_gap_check=False),
  S("nm_cutin_crawl_8m_stop", [("hold", 2.5), ("acc", -2.5, 0.0), ("hold", 6.0)], ev0=8 * MPH, lv0=8 * MPH,
    cut=dict(t=2.0, gap=8.0, v=6 * MPH), kind="nearmiss", set_gap_check=False),
  # --- repeated roll-stop (Matt, real world: "rolling stop roll stop resulted in my car
  # getting closer each time"). >= 6 creep-stop cycles; standstill gap after every stop.
  *[S(f"rs_{d}m", rollstop([d] * 8), kind="rollstop") for d in (0.5, 1, 2, 3, 5)],
  S("rs_mixed", rollstop([0.5, 1, 2, 3, 5, 1, 0.5, 2, 3]), kind="rollstop"),
  S("rs_mixed_noisy", rollstop([0.5, 1, 2, 3, 5, 1, 0.5, 2, 3]), kind="rollstop", noise=True),
  S("rs_1m_noisy", rollstop([1] * 8), kind="rollstop", noise=True),
  S("rs_3m_fast_pause", rollstop([3] * 8, pauses=(0.5,)), kind="rollstop"),
  # long pauses: ego reaches standstill in every cycle
  *[S(f"rs_{d}m_3s", rollstop([d] * 8, pauses=(3.0,)), kind="rollstop") for d in (0.5, 1, 2, 5)],
  S("rs_2m_3s_noisy", rollstop([2] * 8, pauses=(3.0,)), kind="rollstop", noise=True),
  # --- dev-c4-test2 lead matching (accel / brake tracking, soft cut-in recovery)
  *[S(f"lm_accel_{a}", [("hold", 8.0), ("acc", a, 40 * MPH), ("hold", 6.0)], ev0=15 * MPH, lv0=15 * MPH,
      v_cruise=50 * MPH, kind="leadmatch", t_tail=2.0) for a in (0.5, 1.0, 1.5)],
  *[S(f"lm_brake_{a}", [("hold", 8.0), ("acc", -a, 10 * MPH), ("hold", 8.0)], ev0=35 * MPH, lv0=35 * MPH,
      v_cruise=45 * MPH, kind="leadmatch", t_tail=2.0) for a in (1, 2, 3)],
  *[S(f"lm_cutin_{m}mph_{g}m", [("hold", 12.0)], ev0=m * MPH, lv0=m * MPH, v_cruise=(m + 5) * MPH,
      cut=dict(t=4.0, gap=float(g), v=(m - 1) * MPH), kind="cutin", set_gap_check=False)
    for m in (15, 30, 45) for g in (5, 10, 15)],
]


class Lead:
  def __init__(self, segs, v0):
    self.segs = list(segs); self.v = v0; self.a = 0.0; self.i = 0; self.t_seg = 0.0; self.d_seg = 0.0
    self.started = None  # first time it moves from a stop

  def step(self, t, dt):
    a = 0.0
    while self.i < len(self.segs):
      seg = self.segs[self.i]
      if seg[0] == "hold":
        if self.t_seg < seg[1]:
          break
      elif seg[0] == "acc":
        _, acc, vt = seg
        if (acc > 0 and self.v < vt - 1e-6) or (acc < 0 and self.v > vt + 1e-6):
          a = acc; break
      elif seg[0] == "dist":
        _, au, ad, D = seg
        vp = math.sqrt(2 * D / (1 / au + 1 / abs(ad)))
        if not hasattr(self, "_peak") or self._peak_i != self.i:
          self._peak = False; self._peak_i = self.i
        if not self._peak and self.v < vp:
          a = au; break
        self._peak = True
        if self.v > 1e-6:
          a = ad; break
      self.i += 1; self.t_seg = 0.0
    self.a = a
    v_new = self.v + a * dt
    for seg in self.segs[self.i:self.i + 1]:
      if seg[0] == "acc":
        v_new = min(v_new, seg[2]) if seg[1] > 0 else max(v_new, seg[2])
    v_new = max(v_new, 0.0)
    if self.started is None and self.v <= 1e-6 and v_new > 1e-6:
      self.started = t
    self.v = v_new
    self.t_seg += dt
    return self.v

  def done(self):
    return self.i >= len(self.segs)


def SETGAP(p):
  import inspect
  return {"set_gap": stop_gap(p)} if "set_gap" in inspect.signature(HC.HumanCity.update).parameters else {}


def run(personality, sc, cand):
  exp = False
  cfg = cand.get("human")
  hc = HC.HumanCity(HC.HumanCityConfig(**cfg)) if cfg is not None else None
  hold = HC.LeadHoldover() if (hc is not None and hc.config.lead_holdover) else None
  use_rel = bool(cand.get("release", False) and release_prev_accel is not None)
  import openpilot.selfdrive.controls.lib.longcontrol as LCM
  if hasattr(LCM, "STOPPING_FOLLOW_PLAN"):
    LCM.STOPPING_FOLLOW_PLAN = bool(cand.get("stopping_follow_plan", use_rel))
  try:
    import opendbc.sunnypilot.car.toyota.launch_release as LR
  except Exception:
    LR = None
  if LR is not None and hasattr(LR, "LAUNCH_KICK_ACCEL"):
    LR.LAUNCH_KICK_ACCEL = float(cand.get("kick", LR.LAUNCH_KICK_ACCEL)) if use_rel else 0.0
  if LR is not None and hasattr(LR, "LOW_SPEED_WINDDOWN_RATE"):
    LR.LOW_SPEED_WINDDOWN_RATE = float(cand.get("winddown", LR.LOW_SPEED_WINDDOWN_RATE)) if use_rel else 4.0
  use_drop = bool(use_rel and LR is not None and hasattr(LR, "drop_positive_on_brake") and cand.get("drop", True))
  if use_rel and cand.get("release_hold") is not None:
    import opendbc.sunnypilot.car.toyota.launch_release as LR
    LR.LAUNCH_RELEASE_HOLD = float(cand["release_hold"])
  mpc = LM.LongitudinalMpc()
  g_stop = stop_gap(personality)
  ev = sc["ev0"]; lv = sc["lv0"]
  gap0 = sc["gap0"] if sc["gap0"] is not None else (g_stop if lv < 0.1 else eq_gap(personality, lv))
  lead = Lead(sc["segs"], lv)
  cut = sc["cut"]
  CP = SimpleNamespace(stopAccel=-2.0, vEgoStopping=0.25, vEgoStarting=0.25,
                       longitudinalTuning=SimpleNamespace(kiBP=[0.], kiV=[0.]))
  lc = LongControl(CP, SimpleNamespace(enableGasInterceptor=False))
  pid = PIDController(0.0, ([2., 5.], [0.5, 0.25]), k_f=1.0, pos_limit=2.0, neg_limit=-3.5, rate=1 / 0.03)
  aego_f = FirstOrderFilter(0.0, 0.25, 0.03)
  vdf = FirstOrderFilter(ev, 2.0, 0.05)
  v_cruise = sc["v_cruise"]
  dt_c = 0.01
  ex, ea = 0.0, 0.0
  lx = gap0; la_f = 0.0; tau_f = 1.5
  out_a = 0.0; plan_a = 0.0; ss = ev < 0.3; a_cruise = 0.0; prev_accel = 0.0; sent = 0.0
  delay = [0.0] * 5
  def visible(t):
    if sc["appear"] is not None and t < sc["appear"] + 1e-9 and sc["appear"] > 0:
      return False
    return not any(a <= t < b for a, b in sc["lost"])
  if ev > 0.1 and sc["appear"] is None:
    for _ in range(40):
      mpc.set_weights(True, personality=personality, experimental_mode=exp)
      mpc.set_cur_state(ev, 0.0)
      mpc.update(H._radar(gap0, lv, 0.0), personality=personality, experimental_mode=exp)
    a_cruise = 0.0
  cut_done = False
  solve_fail = 0
  rng = np.random.default_rng(1234)
  t = 0.0; k = 0
  min_gap = 1e9; min_gap_rel_set = 1e9; min_gap_stop_rel = 1e9; peak_a = 0.0; peak_d = 0.0; peak_j = 0.0; peak_j_low = 0.0
  jerk_sq = 0.0; n_j = 0
  ea_hist = []; v_hist = []; collided = False; v_peak = 0.0
  moves = []; lead_was_stopped = lv < 0.05; t_lead_start = None
  stop_gaps = []; ego_stopped_prev = ev < 0.05; cycle_gaps = []
  hw_samples = []; dense_max = 0.0; trace = []
  first_brake_t = None; lead_brake_t = None
  t_end_lead = None
  t_max = 240.0
  while t < t_max:
    lv = lead.step(t, dt_c)
    la_true = lead.a
    if la_true < -0.5 and lead_brake_t is None and lv > 0.1:
      lead_brake_t = t
    if cut and not cut_done and t >= cut["t"]:
      cut_done = True
      lx = ex + cut["gap"]; lv = cut["v"]; lead.v = lv; la_f = 0.0
    lx += lv * dt_c
    la_f += (la_true - la_f) * (1 - math.exp(-dt_c / 0.3))
    gap = lx - ex
    if lead.done() and t_end_lead is None:
      t_end_lead = t
    if t_end_lead is not None and t > t_end_lead + sc["t_tail"]:
      break
    if lv < 0.05:
      lead_was_stopped = True
    elif lead_was_stopped and lv > 0.05:
      lead_was_stopped = False
      cycle_gaps.append([round(gap, 2), round(ev, 2)])   # gap / ego speed when the lead moves again
      if ev < 0.05:
        t_lead_start = t
    if t_lead_start is not None and ev > 0.1:
      moves.append(round(t - t_lead_start, 3)); t_lead_start = None
    if k % 5 == 0:
      if abs(la_f) < 0.5:
        tau_f = 1.5
      else:
        tau_f += (0.0 - tau_f) * (0.05 / (0.45 + 0.05))
      vis = visible(t)
      if sc["noise"]:
        rs = H._radar(gap + rng.normal(0, 0.15), max(lv + rng.normal(0, 0.15), 0.0), la_f + rng.normal(0, 0.3),
                      tau=tau_f, present=vis)
      else:
        rs = H._radar(gap, lv, la_f, tau=tau_f, present=vis)
      if vis:
        rs.leadOne.vRel = float(lv - ev)
      standstill = ev < 1e-3
      cap = float(LP.get_max_accel(ev, personality == P.aggressive, personality == P.relaxed))
      vdf.x = max(0.0, vdf.update(ev))
      pac = not standstill
      fcw_prev = mpc.crash_cnt > 2 and not standstill
      jf = None; tf = None; dcost = None; lj = None; lgo = False; csoft = False; mpc.stop_extra_predicted = 0.0
      mpc.launch_margin = 0.0
      mpc.lead_danger_factor = LM.LEAD_DANGER_FACTOR
      if hold is not None:
        held = hold.update(rs.leadOne, ev, 0.05)
        if held is not rs.leadOne:
          rs = SimpleNamespace(leadOne=held, leadTwo=rs.leadTwo)
      if hc is not None:
        base_t = LM.get_T_FOLLOW(personality, exp, vdf.x)
        o = hc.update(ev, rs.leadOne, PNAME[personality], LM.get_jerk_factor(personality, exp), base_t, cap,
                      follow_gap(ev, float(rs.leadOne.vLead), base_t) if HC.lead_present(rs.leadOne) else float("nan"), fcw_prev, 0.05,
                      plan_a_min=float(np.min(mpc.a_solution[:HUMAN_PLAN_N])), **SETGAP(personality))
        jf, cap, tf, dcost = o.jerk_factor, o.cap, o.t_follow, o.danger_cost
        lj = getattr(o, "launch_jerk", None)
        lgo = bool(getattr(o, "launch_go", False))
        mpc.launch_margin = float(getattr(o, "launch_margin", 0.0))
        if os.environ.get("HCSIM_DEBUG") and k % 5 == 0:
          print(f"DBG t={t:5.2f} d={float(rs.leadOne.dRel):6.2f} vl={float(rs.leadOne.vLead):5.2f} ev={ev:5.2f} "
                f"lm={mpc.launch_margin:4.2f} ep={hc.launch_ep} w={hc.launch_w:4.2f} anc={getattr(hc, 'stop_anchor', None)}", file=sys.stderr)
        csoft = bool(getattr(o, "cutin_soft", False))
        if getattr(o, "mpc_lead", None) is not None:
          rs = SimpleNamespace(leadOne=o.mpc_lead, leadTwo=rs.leadTwo)
        mpc.stop_extra_predicted = o.stop_extra_predicted
        if o.danger_factor is not None:
          mpc.lead_danger_factor = o.danger_factor
        dense_max = max(dense_max, o.dense)
      mpc.set_weights(pac, personality=personality, experimental_mode=exp, jerk_factor=jf, danger_cost=dcost)
      mpc.set_cur_state(vdf.x, out_a)
      mpc.update(rs, personality=personality, experimental_mode=exp, t_follow=tf)
      if mpc.solution_status != 0:
        solve_fail += 1
      vt = np.interp(CTRL_T, LM.T_IDXS, mpc.v_solution)
      at = np.interp(CTRL_T, LM.T_IDXS, mpc.a_solution)
      a_prev = out_a
      a_mpc = get_accel_from_plan(vt, at, CTRL_T, action_t=0.05 + 0.05)
      a_cruise = LP.get_cruise_accel(v_cruise, ev, a_cruise, 0.05, cap=cap)
      ss_mpc = should_stop(ev, a_mpc)
      if lgo:
        ss_mpc = bool(ev < HC.LAUNCH_GO_V_MAX and a_mpc < HC.LAUNCH_GO_A_MIN)
      ss = ss_mpc or should_stop(ev, a_cruise)
      a_t = float(np.clip(min(a_mpc, a_cruise), ACCEL_MIN, cap))
      try:
        a_t = float(LP.limit_accel_rise(a_t, a_prev, ev, 0.05, jerk=lj))
      except TypeError:  # older trees without the jerk override
        a_t = float(LP.limit_accel_rise(a_t, a_prev, ev, 0.05))
      if csoft:
        a_t = float(HC.cutin_soft_limit(a_t, a_prev, 0.05))
      out_a = a_t; plan_a = a_t
      vdf.x = vdf.x + 0.05 * (out_a + a_prev) / 2.0
    CS = SimpleNamespace(vEgo=ev, aEgo=ea, brakePressed=False, standstill=ev < 1e-3,
                         cruiseState=SimpleNamespace(standstill=False))
    act = float(lc.update(True, CS, plan_a, ss, (ACCEL_MIN, ACCEL_MAX)))
    if k % 3 == 0:
      if use_rel:
        prev_accel = release_prev_accel(prev_accel, act, ev, lc.long_control_state == LongCtrlState.pid, True)
      if use_drop:
        prev_accel = LR.drop_positive_on_brake(prev_accel, act, ev, True)
      wd = LR.winddown_limit(ev, -4.0 * 0.03, 0.03) if (use_rel and LR is not None and hasattr(LR, "winddown_limit")) else -4.0 * 0.03
      pcm = rate_limit(act, prev_accel, wd, 4.0 * 0.03)
      if use_rel and LR is not None and hasattr(LR, "launch_kick_floor"):
        kf = LR.launch_kick_floor(act, ev, lc.long_control_state == LongCtrlState.pid, True)
        if kf is not None:
          pcm = max(pcm, kf)
      prev_accel = pcm
      prev_ae = aego_f.x; aego_f.update(ea)
      j_ego = (aego_f.x - prev_ae) / 0.03
      a_fut = ea + j_ego * float(np.interp(ev, [2., 5.], [0.25, 0.5]))
      pid.i -= 0.03 * 0.03 * float(np.sign(pid.i))
      pcm = pid.update(pcm - a_fut, speed=ev, feedforward=pcm,
                       freeze_integrator=lc.long_control_state != LongCtrlState.pid)
      sent = float(np.clip(pcm, -3.5, 2.0))
    delay.append(sent); d = delay.pop(0)
    ea = ea + (d - ea) * (1 - math.exp(-dt_c / 0.25))
    ea_eff = 0.0 if (ev <= 0.0 and ea < 0.0) else ea
    ev = max(0.0, ev + ea_eff * dt_c)
    ex += ev * dt_c
    gap = lx - ex
    if lead_brake_t is not None and first_brake_t is None and ea_eff < -0.5:
      first_brake_t = t
    if not (cut and not cut_done):
      min_gap = min(min_gap, gap)
      if lv < 0.05 and ev < 0.05:
        min_gap_stop_rel = min(min_gap_stop_rel, gap - g_stop)
    ego_stopped = ev < 0.01
    if ego_stopped and not ego_stopped_prev and lv < 0.05:
      stop_gaps.append(round(gap, 2))
    ego_stopped_prev = ego_stopped
    if gap <= 0.0: collided = True
    v_peak = max(v_peak, ev)
    peak_a = max(peak_a, ea_eff); peak_d = min(peak_d, ea_eff)
    ea_hist.append(ea_eff); v_hist.append(ev)
    if len(v_hist) > 6: v_hist.pop(0)
    if len(ea_hist) > 5:
      j = (ea_hist[-1] - ea_hist[-6]) / 0.05
      ea_hist.pop(0)
      if ev > 0.1 and min(v_hist) > 0.1:
        peak_j = max(peak_j, abs(j)); jerk_sq += j * j; n_j += 1
      elif ev > 0.0 or ea_hist[-2] > 0.0:
        peak_j_low = max(peak_j_low, abs(j))
    if sc["kind"] == "headway" and t > 35.0 and ev > 1.0:
      hw_samples.append(gap / ev)
    if k % 20 == 0:
      t_used = tf if tf is not None else float(LM.get_T_FOLLOW(personality, exp, ev))
      trace.append([round(t, 2), round(gap, 2), round(lv / MPH, 2), round(ev / MPH, 2), round(plan_a, 3), round(ea_eff, 3),
                    round(la_true, 3), round(follow_gap(ev, lv, t_used), 2), round(t_used, 3)])
    t += dt_c; k += 1
  return dict(collision=collided, min_gap_m=round(min_gap, 2), stop_gap_m=round(g_stop, 2),
              min_gap_minus_stop_gap_m=round(min_gap - g_stop, 2),
              standstill_gap_minus_stop_gap_m=None if min_gap_stop_rel > 1e8 else round(min_gap_stop_rel, 2),
              stop_gaps_m=stop_gaps, cycle_gaps=cycle_gaps, final_gap_m=round(lx - ex, 2), peak_accel=round(peak_a, 3),
              peak_decel=round(peak_d, 3), peak_jerk=round(peak_j, 2), peak_jerk_low=round(peak_j_low, 2), rms_jerk=round(math.sqrt(jerk_sq / max(n_j, 1)), 3),
              t_move=moves, t_move_mean=round(float(np.mean(moves)), 3) if moves else None,
              t_move_max=round(float(max(moves)), 3) if moves else None,
              headway_s=round(float(np.mean(hw_samples)), 3) if hw_samples else None,
              brake_delay_s=None if (first_brake_t is None or lead_brake_t is None) else round(first_brake_t - lead_brake_t, 2),
              v_ego_peak_mph=round(v_peak / MPH, 2), dense_max_s=round(dense_max, 3),
              duration_s=round(t, 1), solve_fail=solve_fail, kind=sc["kind"], set_gap_check=sc["set_gap_check"],
              trace=trace)


pers = [P.aggressive, P.standard, P.relaxed]
only = set(sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] else None
subs = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else None
res = {"tip": TIP.name, "candidate": CAND, "cases": {}}
for p in pers:
  if only and PNAME[p] not in only:
    continue
  res["cases"][PNAME[p]] = {}
  for sc in SCENARIOS:
    if subs and not any(s in sc["name"] for s in subs):
      continue
    r = run(p, sc, CAND)
    res["cases"][PNAME[p]][sc["name"]] = r
    print(f"{CAND.get('name','?'):8s} {PNAME[p]:10s} {sc['name']:36s} coll={int(r['collision'])} "
          f"min={r['min_gap_m']:6.2f}(set {r['stop_gap_m']}) sgap={r['standstill_gap_minus_stop_gap_m']} "
          f"hw={r['headway_s']} a+={r['peak_accel']:.2f} a-={r['peak_decel']:.2f} jpk={r['peak_jerk']:.2f} "
          f"jrms={r['rms_jerk']:.3f} tmove={r['t_move_mean']} bd={r['brake_delay_s']} fail={r['solve_fail']} "
          f"dense={r['dense_max_s']}", flush=True)
OUT.write_text(json.dumps(res, indent=1))
