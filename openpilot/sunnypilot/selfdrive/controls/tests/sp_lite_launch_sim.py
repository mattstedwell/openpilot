#!/usr/bin/env python3
"""dev-c4-sunnypilot-lite closed-loop host sim: stock sunnypilot dev vs lite.

usage: sp_lite_launch_sim.py TREE_ROOT BUILD_DIR OUT_JSON CAND_JSON [personalities] [name-substrings]
CAND_JSON: {"name": str, "extra": bool, "release": bool, "launch": bool}
  extra   -> long_mpc LEAD_STOP_EXTRA (+10 ft standstill gap); False sets it to 0.0 (stock)
  release -> Toyota brake-hold release (opendbc sunnypilot launch_release)
  launch  -> lite_launch planner gates (quick start after lead moves)

Real LongitudinalMpc (host acados build of this tree), the planner's
MPC + cruise path wired like longitudinal_planner.py, real LongControl, and
the Toyota TSS2 carcontroller long path emulated (4 m/s^3 rate limit, TSS2
PID, a_ego_future). Radar: aLeadK lags the true accel by 0.3 s, aLeadTau
follows radard. Plant: 0.05 s delay + 0.25 s lag. NOT simulated: the
driving model's e2e desiredAcceleration / shouldStop candidate (treated as
non-limiting), radard Kalman lag on vLead, the Toyota PCM's own resume
delay from standstill, road grade. Host-only; NOT road-tested.
"""
import json, math, sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sp_lite_host_mpc as HM  # noqa: E402

TREE = Path(sys.argv[1]).resolve()
BUILD = Path(sys.argv[2])
OUT = Path(sys.argv[3])
CAND = json.loads(sys.argv[4])
HM.setup(BUILD, TREE)

import numpy as np  # noqa: E402
from openpilot.cereal import log  # noqa: E402
from opendbc.car.structs import car  # noqa: E402
import openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc as LM  # noqa: E402
import openpilot.selfdrive.controls.lib.longitudinal_planner as LP  # noqa: E402
import openpilot.selfdrive.controls.lib.longcontrol as LCmod  # noqa: E402
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop  # noqa: E402
from openpilot.selfdrive.modeld.constants import ModelConstants  # noqa: E402
from opendbc.car.common.pid import PIDController  # noqa: E402
from opendbc.car.common.filter_simple import FirstOrderFilter  # noqa: E402
from opendbc.car import rate_limit  # noqa: E402
try:
  from opendbc.sunnypilot.car.toyota.launch_release import release_prev_accel
except Exception:
  release_prev_accel = None
try:
  from openpilot.sunnypilot.selfdrive.controls.lib import lite_launch as LL
except Exception:
  LL = None

import os
TRACE = [float(x) for x in os.environ['SIM_TRACE'].split(',')] if os.environ.get('SIM_TRACE') else None
if not CAND.get("extra", False):
  LM.LEAD_STOP_EXTRA = 0.0
USE_LAUNCH = bool(CAND.get("launch", False) and LL is not None)
if LL is not None:
  LL.ENABLED = USE_LAUNCH
  for _k, _v in CAND.get("ll", {}).items():  # exploratory overrides only
    setattr(LL, _k, _v)
if CAND.get("rel_hold") is not None:
  import opendbc.sunnypilot.car.toyota.launch_release as _LR
  _LR.LAUNCH_RELEASE_HOLD = float(CAND["rel_hold"])

P = log.LongitudinalPersonality
PNAME = {P.aggressive: "aggressive", P.standard: "standard", P.relaxed: "relaxed"}
LongCtrlState = car.CarControl.Actuators.LongControlState
CTRL_T = ModelConstants.T_IDXS[:CONTROL_N]
ACCEL_MIN, ACCEL_MAX = -3.5, 2.0
MPH = 0.44704


def stop_target():
  return LM.STOP_DISTANCE + float(LM.LEAD_STOP_EXTRA)


def eq_gap(p, v):
  return float(LM.get_T_FOLLOW(p) * v + LM.STOP_DISTANCE)


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


def roll_stop(n, D, au, ad, hold):
  return [("hold", 3.0)] + sum([[("dist", au, ad, D), ("hold", hold)] for _ in range(n)], [])


SCENARIOS = [
  *[S(f"follow_{m}mph", [("hold", 45.0)], ev0=m * MPH, lv0=m * MPH, kind="headway", headway_mph=m, t_tail=0.0,
      v_cruise=(m + 5) * MPH) for m in (15, 25, 35, 45)],
  S("signal_queue_40mph", signal_cycle(40, up=1.5, dn=-1.8)),
  S("signal_queue_35mph_firm", signal_cycle(35, up=2.0, dn=-2.5)),
  S("approach_stopped_queue_40mph", [("hold", 1.0)], ev0=40 * MPH, lv0=0.0, gap0=120.0, t_tail=25.0),
  S("approach_stopped_queue_45mph", [("hold", 1.0)], ev0=45 * MPH, lv0=0.0, gap0=140.0, t_tail=25.0),
  S("lead_slows_to_stop_40mph", [("hold", 4.0), ("acc", -1.5, 0.0), ("hold", 8.0)], ev0=40 * MPH, lv0=40 * MPH),
  S("lead_hard_brake_40mph_3.5", [("hold", 4.0), ("acc", -3.5, 0.0), ("hold", 8.0)], ev0=40 * MPH, lv0=40 * MPH),
  S("lead_hard_brake_40mph_4.5", [("hold", 4.0), ("acc", -4.5, 0.0), ("hold", 8.0)], ev0=40 * MPH, lv0=40 * MPH,
    kind="beyond_limit"),
  S("lead_hard_brake_25mph", [("hold", 4.0), ("acc", -4.0, 0.0), ("hold", 8.0)], ev0=25 * MPH, lv0=25 * MPH),
  S("accordion_0_15mph", accordion([8, 15, 12, 5, 15], [1.0, 1.5, 1.2, 0.8, 2.0], [-1.5, -2.5, -2.0, -1.5, -3.0])),
  S("accordion_noisy_radar", accordion([10, 15, 6, 12], [1.2, 1.5, 0.8, 1.0], [-2.0, -2.5, -1.5, -3.0]), noise=True),
  S("rolling_5_25mph", [("hold", 3.0)] + sum([[("acc", 1.0, 25 * MPH), ("hold", 2.0), ("acc", -1.5, 5 * MPH), ("hold", 1.0)]
                                              for _ in range(4)], []), ev0=5 * MPH, lv0=5 * MPH),
  S("launch_to35mph", [("hold", 2.0), ("acc", 1.5, 35 * MPH), ("hold", 10.0)], kind="launch"),
  S("launch_gentle_0.5", [("hold", 2.0), ("acc", 0.5, 20 * MPH), ("hold", 6.0)], kind="launch"),
  S("launch_brisk_2.0", [("hold", 2.0), ("acc", 2.0, 30 * MPH), ("hold", 6.0)], kind="launch"),
  S("stopped_lead_hold", [("hold", 15.0)]),
  S("cutin_30mph_10m", [("hold", 12.0)], ev0=30 * MPH, lv0=30 * MPH, cut=dict(t=3.0, gap=10.0, v=28 * MPH),
    set_gap_check=False),
  S("cutin_12mph_then_stop", [("hold", 4.5), ("acc", -1.5, 0.0), ("hold", 6.0)], ev0=12 * MPH, lv0=12 * MPH,
    cut=dict(t=3.0, gap=7.0, v=10 * MPH), set_gap_check=False),
  # Nov 13 near-miss family
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
  # Matt's real-world failure: roll, stop, roll, stop ... gap must not shrink cycle to cycle
  S("rollstop_1m_x8", roll_stop(8, 1.0, 1.0, -1.5, 1.5), kind="rollstop"),
  S("rollstop_2m_x8", roll_stop(8, 2.0, 1.0, -2.0, 1.5), kind="rollstop"),
  S("rollstop_3m_x8", roll_stop(8, 3.0, 1.2, -2.5, 2.0), kind="rollstop"),
  S("rollstop_2m_short_hold_x8", roll_stop(8, 2.0, 1.5, -3.0, 0.5), kind="rollstop"),
  S("rollstop_0.5m_x10", roll_stop(10, 0.5, 0.8, -1.5, 1.0), kind="rollstop"),
  S("rollstop_5m_x6", roll_stop(6, 5.0, 1.0, -2.0, 3.0), kind="rollstop"),
  S("rollstop_2m_noisy_x8", roll_stop(8, 2.0, 1.0, -2.0, 1.5), noise=True, kind="rollstop"),
]


class Lead:
  def __init__(self, segs, v0):
    self.segs = list(segs); self.v = v0; self.a = 0.0; self.i = 0; self.t_seg = 0.0
    self._peak = False; self._peak_i = -1

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
        if self._peak_i != self.i:
          self._peak = False; self._peak_i = self.i
        if not self._peak and self.v < vp:
          a = au; break
        self._peak = True
        if self.v > 1e-6:
          a = ad; break
      self.i += 1; self.t_seg = 0.0
    self.a = a
    v_new = self.v + a * dt
    seg = self.segs[self.i] if self.i < len(self.segs) else None
    if seg is not None and seg[0] == "acc":
      v_new = min(v_new, seg[2]) if seg[1] > 0 else max(v_new, seg[2])
    self.v = max(v_new, 0.0)
    self.t_seg += dt
    return self.v

  def done(self):
    return self.i >= len(self.segs)


def run(personality, sc):
  use_rel = bool(CAND.get("release", False) and release_prev_accel is not None)
  mpc = LM.LongitudinalMpc()
  g_stop = stop_target()
  ev = sc["ev0"]; lv = sc["lv0"]
  gap0 = sc["gap0"] if sc["gap0"] is not None else (g_stop if lv < 0.1 else eq_gap(personality, lv))
  lead = Lead(sc["segs"], lv)
  cut = sc["cut"]
  CP = SimpleNamespace(stopAccel=-2.0, vEgoStopping=0.25, vEgoStarting=0.25,
                       longitudinalTuning=SimpleNamespace(kiBP=[0.], kiV=[0.]))
  lc = LCmod.LongControl(CP, SimpleNamespace(enableGasInterceptor=False))
  launch = LL.LiteLaunch() if USE_LAUNCH else None
  pid = PIDController(0.0, ([2., 5.], [0.5, 0.25]), k_f=1.0, pos_limit=2.0, neg_limit=-3.5, rate=1 / 0.03)
  aego_f = FirstOrderFilter(0.0, 0.25, 0.03)
  vdf = FirstOrderFilter(ev, 2.0, 0.05)
  v_cruise = sc["v_cruise"]
  dt_c = 0.01
  ex, ea = 0.0, 0.0
  lx = gap0; la_f = 0.0; tau_f = 1.5
  out_a = 0.0; plan_a = 0.0; ss = ev < 0.3; a_cruise = 0.0; prev_accel = 0.0; sent = 0.0
  quick = False
  delay = [0.0] * 5

  def visible(t):
    if sc["appear"] is not None and t < sc["appear"] + 1e-9 and sc["appear"] > 0:
      return False
    return not any(a <= t < b for a, b in sc["lost"])
  if ev > 0.1 and sc["appear"] is None:
    for _ in range(40):
      mpc.set_weights(True, personality=personality)
      mpc.set_cur_state(ev, 0.0)
      mpc.update(HM.radar(gap0, lv, 0.0), personality=personality)
  cut_done = False; solve_fail = 0
  rng = np.random.default_rng(1234)
  t = 0.0; k = 0
  min_gap = 1e9; min_gap_stop_rel = 1e9; peak_a = 0.0; peak_d = 0.0; peak_j = 0.0; jerk_sq = 0.0; n_j = 0
  ea_hist = []; v_hist = []; collided = False
  firsts = []; t_first_pending = None
  moves = []; lead_was_stopped = lv < 0.05; t_lead_start = None
  hold_end = []
  stop_gaps = []; ego_stopped_prev = ev < 0.05
  hw_samples = []; t_end_lead = None; ego_moved_while_lead_stopped = 0.0
  while t < 300.0:
    lv = lead.step(t, dt_c)
    la_true = lead.a
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
      if t > 0.5:
        hold_end.append((round(lx - ex, 2), round(ev / MPH, 2)))
      if ev < 0.05:
        t_lead_start = t
    if t_lead_start is not None and ev > 0.01 and t_first_pending is None:
      t_first_pending = t_lead_start
      firsts.append(round(t - t_lead_start, 3))
    if t_lead_start is not None and ev > 0.1:
      moves.append(round(t - t_lead_start, 3)); t_lead_start = None; t_first_pending = None
    if t_lead_start is not None and lv < 0.01 and ev < 0.01:
      t_lead_start = None; t_first_pending = None  # lead stopped again before we moved: not a launch sample
    if k % 5 == 0:
      if abs(la_f) < 0.5:
        tau_f = 1.5
      else:
        tau_f += (0.0 - tau_f) * (0.05 / (0.45 + 0.05))
      vis = visible(t)
      if sc["noise"]:
        rs = HM.radar(gap + rng.normal(0, 0.15), max(lv + rng.normal(0, 0.15), 0.0), la_f + rng.normal(0, 0.3),
                      tau=tau_f, present=vis)
      else:
        rs = HM.radar(gap, lv, la_f, tau=tau_f, present=vis)
      if vis:
        rs.leadOne.vRel = float(lv - ev)
      standstill = ev < 1e-3
      vdf.x = max(0.0, vdf.update(ev))
      mpc.set_weights(not standstill, personality=personality)
      mpc.set_cur_state(vdf.x, out_a)
      mpc.update(rs, personality=personality)
      if mpc.solution_status != 0:
        solve_fail += 1
      fcw = mpc.crash_cnt > 2 and not standstill
      vt = np.interp(CTRL_T, LM.T_IDXS, mpc.v_solution)
      at = np.interp(CTRL_T, LM.T_IDXS, mpc.a_solution)
      a_prev = out_a
      a_mpc = get_accel_from_plan(vt, at, CTRL_T, action_t=0.05 + 0.05)
      ss_mpc = should_stop(ev, a_mpc)
      a_cruise = LP.get_cruise_accel(v_cruise, ev, a_cruise, 0.05)
      ss_cruise = should_stop(ev, a_cruise)
      a_t = float(np.clip(min(a_mpc, a_cruise), ACCEL_MIN, ACCEL_MAX))
      ss = ss_mpc or ss_cruise
      if launch is not None:
        quick, a_t = launch.update(rs.leadOne, ev, standstill, fcw, a_t, None, False)
        if quick:
          ss = False
      out_a = a_t; plan_a = a_t
      vdf.x = vdf.x + 0.05 * (out_a + a_prev) / 2.0
    CS = SimpleNamespace(vEgo=ev, aEgo=ea, brakePressed=False, standstill=ev < 1e-3,
                         cruiseState=SimpleNamespace(standstill=False))
    act = float(lc.update(True, CS, plan_a, ss, (ACCEL_MIN, ACCEL_MAX)))
    if k % 3 == 0:
      if use_rel:
        prev_accel = release_prev_accel(prev_accel, act, ev, lc.long_control_state == LongCtrlState.pid, True)
      pcm = rate_limit(act, prev_accel, -4.0 * 0.03, 4.0 * 0.03)
      prev_accel = pcm
      prev_ae = aego_f.x; aego_f.update(ea)
      j_ego = (aego_f.x - prev_ae) / 0.03
      a_fut = ea + j_ego * float(np.interp(ev, [2., 5.], [0.25, 0.5]))
      pid.i -= 0.03 * 0.03 * float(np.sign(pid.i))
      pcm = pid.update(pcm - a_fut, speed=ev, feedforward=pcm,
                       freeze_integrator=lc.long_control_state != LongCtrlState.pid)
      sent = float(np.clip(pcm, -3.5, 2.0))
    if TRACE and TRACE[0] <= t <= TRACE[1] and k % 2 == 0:
      print(f"TR t={t:5.2f} lv={lv:5.2f} gap={lx-ex:6.2f} a_plan={plan_a:6.3f} ss={int(ss)} st={int(lc.long_control_state)} "
            f"act={act:6.3f} prev={prev_accel:6.3f} sent={sent:6.3f} ea={ea:6.3f} ev={ev:5.3f} q={int(quick)}")
    delay.append(sent); d = delay.pop(0)
    ea = ea + (d - ea) * (1 - math.exp(-dt_c / 0.25))
    ea_eff = 0.0 if (ev <= 0.0 and ea < 0.0) else ea
    ev = max(0.0, ev + ea_eff * dt_c)
    ex += ev * dt_c
    gap = lx - ex
    if lv < 0.01 and ev > 0.05 and gap < g_stop:
      ego_moved_while_lead_stopped = max(ego_moved_while_lead_stopped, ev)
    if not (cut and not cut_done):
      min_gap = min(min_gap, gap)
      if lv < 0.05 and ev < 0.05:
        min_gap_stop_rel = min(min_gap_stop_rel, gap - g_stop)
    ego_stopped = ev < 0.01
    if ego_stopped and not ego_stopped_prev and lv < 0.05:
      stop_gaps.append(round(gap, 2))
    ego_stopped_prev = ego_stopped
    if gap <= 0.0:
      collided = True
    peak_a = max(peak_a, ea_eff); peak_d = min(peak_d, ea_eff)
    ea_hist.append(ea_eff); v_hist.append(ev)
    if len(v_hist) > 6:
      v_hist.pop(0)
    if len(ea_hist) > 5:
      j = (ea_hist[-1] - ea_hist[-6]) / 0.05
      ea_hist.pop(0)
      if ev > 0.1 and min(v_hist) > 0.1:
        peak_j = max(peak_j, abs(j)); jerk_sq += j * j; n_j += 1
    if sc["kind"] == "headway" and t > 35.0 and ev > 1.0:
      hw_samples.append(gap / ev)
    t += dt_c; k += 1
  return dict(collision=collided, min_gap_m=round(min_gap, 2), stop_target_m=round(g_stop, 2),
              min_gap_minus_target_m=round(min_gap - g_stop, 2),
              standstill_gap_minus_target_m=None if min_gap_stop_rel > 1e8 else round(min_gap_stop_rel, 2),
              stop_gaps_m=stop_gaps, hold_end_gap_m=[g for g, _ in hold_end], hold_end_v_mph=[v for _, v in hold_end], final_gap_m=round(lx - ex, 2), peak_accel=round(peak_a, 3),
              peak_decel=round(peak_d, 3), peak_jerk=round(peak_j, 2), rms_jerk=round(math.sqrt(jerk_sq / max(n_j, 1)), 3),
              t_first_motion=firsts, t_move=moves, t_move_mean=round(float(np.mean(moves)), 3) if moves else None,
              t_move_max=round(float(max(moves)), 3) if moves else None,
              headway_s=round(float(np.mean(hw_samples)), 3) if hw_samples else None,
              max_v_inside_target_lead_stopped=round(ego_moved_while_lead_stopped, 3),
              duration_s=round(t, 1), solve_fail=solve_fail, kind=sc["kind"], set_gap_check=sc["set_gap_check"])


only = set(sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] else None
subs = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else None
res = {"tree": str(TREE), "candidate": CAND, "cases": {}}
for p in (P.aggressive, P.standard, P.relaxed):
  if only and PNAME[p] not in only:
    continue
  res["cases"][PNAME[p]] = {}
  for sc in SCENARIOS:
    if subs and not any(s in sc["name"] for s in subs):
      continue
    r = run(p, sc)
    res["cases"][PNAME[p]][sc["name"]] = r
    print(f"{CAND.get('name', '?'):6s} {PNAME[p]:10s} {sc['name']:36s} coll={int(r['collision'])} "
          f"min={r['min_gap_m']:6.2f}(tgt {r['stop_target_m']}) sgap-tgt={r['standstill_gap_minus_target_m']} "
          f"stops={r['stop_gaps_m'][:10]} hold_end={r['hold_end_gap_m'][:10]} hw={r['headway_s']} a+={r['peak_accel']:.2f} a-={r['peak_decel']:.2f} "
          f"jpk={r['peak_jerk']:.2f} t1={r['t_first_motion']} tmove={r['t_move']} fail={r['solve_fail']}", flush=True)
OUT.write_text(json.dumps(res, indent=1))
