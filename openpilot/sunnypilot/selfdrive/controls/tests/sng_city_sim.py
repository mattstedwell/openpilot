#!/usr/bin/env python3
"""dev-c4-sng-test: stop-and-go / city closed-loop sim.

usage: sng_city_sim.py TIP_ROOT BUILD_DIR OUT_JSON CANDIDATE_JSON [personalities] [scenario-substrings]

CANDIDATE_JSON: {"name": ..., "sng": {SngConfig fields} | null, "release": bool}
  sng null  -> no stop-and-go module (dev-c4 9c6e321c06 behavior)
  sng.safe_launch -> t309u SafeLaunch MPC weighting in the planner
  release   -> t309u Toyota brake-hold release in the carcontroller

Runs this tree's real LongitudinalMpc (host acados build), planner
pieces (caps, cruise accel, positive-accel rise limit, SafeLaunch,
StopAndGo) wired the same way longitudinal_planner.py wires them, real
LongControl, and the Toyota TSS2 carcontroller long path emulated
line-for-line (4 m/s³ rate limit, TSS2 PID, a_ego_future). Radar lead:
aLeadK is a 0.3 s lag of the true accel and aLeadTau follows radard
(1.5 when |a| < 0.5, else filtered toward 0 with 0.45 s). Plant: 0.05 s
delay + 0.25 s lag, hybrid actuator delay 0.05 s. The model's e2e
candidate is not simulated (always-Chill build). Host-only; not
road-tested.
"""
import importlib.util, json, math, sys
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
  from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_safe_launch import SafeLaunch, desired_follow_gap, launch_gate_ok
except Exception:
  SafeLaunch = desired_follow_gap = None
try:
  from opendbc.sunnypilot.car.toyota.launch_release import release_prev_accel
except Exception:
  release_prev_accel = None
from openpilot.sunnypilot.selfdrive.controls.lib import devc4_sng as SNG

P = log.LongitudinalPersonality
PNAME = {P.aggressive: "aggressive", P.standard: "standard", P.relaxed: "relaxed"}
LongCtrlState = car.CarControl.Actuators.LongControlState
CTRL_T = ModelConstants.T_IDXS[:CONTROL_N]
ACCEL_MIN, ACCEL_MAX = -3.5, 2.0


SNG_PLAN_N = int(np.searchsorted(LM.T_IDXS, 2.0)) + 1


def stop_gap(p):
  return LM.STOP_DISTANCE + LM.get_lead_stop_extra(p, False)


def eq_gap(p, v):
  return float(LM.get_T_FOLLOW(p, False, v) * v + LM.STOP_DISTANCE)


# --- scenarios -------------------------------------------------------------
# A scenario is: ego v0, lead v0, initial gap (None -> equilibrium/standstill
# gap), v_cruise, list of lead segments (kind, value, arg), optional cut-in.
#   ("acc", a, v_target)  accelerate/brake at a until v_target
#   ("hold", seconds)     hold speed
#   ("dist", a_up, a_dn, D) creep D metres then stop
def S(name, segs, ev0=0.0, lv0=0.0, gap0=None, v_cruise=40 * MPH, cut=None, t_tail=10.0, from_stop=True, noise=False):
  return dict(name=name, segs=segs, ev0=ev0, lv0=lv0, gap0=gap0, v_cruise=v_cruise, cut=cut, t_tail=t_tail,
              from_stop=from_stop, noise=noise)


def accordion(peaks, ups, dns, wait=4.0, hold=2.0):
  segs = [("hold", 2.0)]
  for vp, au, ad in zip(peaks, ups, dns):
    segs += [("acc", au, vp * MPH), ("hold", hold), ("acc", ad, 0.0), ("hold", wait)]
  return segs


SCENARIOS = [
  S("accordion_0_15mph", accordion([8, 15, 12, 5, 15], [1.0, 1.5, 1.2, 0.8, 2.0], [-1.5, -2.5, -2.0, -1.5, -3.0])),
  S("accordion_noisy_radar", accordion([10, 15, 6, 12], [1.2, 1.5, 0.8, 1.0], [-2.0, -2.5, -1.5, -3.0]), noise=True),
  S("rolling_5_15mph", [("hold", 3.0)] + sum([[("acc", 0.8, 15 * MPH), ("hold", 2.0), ("acc", -1.2, 5 * MPH), ("hold", 1.0)]
                                              for _ in range(5)], []), ev0=5 * MPH, lv0=5 * MPH, from_stop=False),
  S("creeps_1.5_3_1m", [("hold", 2.0), ("dist", 1.0, -1.5, 1.5), ("hold", 3.0), ("dist", 1.0, -3.0, 3.0), ("hold", 3.0),
                        ("dist", 0.5, -1.5, 1.0), ("hold", 4.0)]),
  S("launch_to25mph", [("hold", 2.0), ("acc", 1.0, 25 * MPH), ("hold", 8.0)]),
  S("launch_fast_to25mph", [("hold", 2.0), ("acc", 2.0, 25 * MPH), ("hold", 8.0)]),
  S("stopped_lead_hold", [("hold", 15.0)]),
  S("false_start", [("hold", 2.0), ("acc", 1.0, 4 * MPH), ("acc", -4.0, 0.0), ("hold", 6.0)]),
  S("abrupt_stop_from15", [("hold", 4.0), ("acc", -4.0, 0.0), ("hold", 6.0)], ev0=15 * MPH, lv0=15 * MPH, from_stop=False),
  S("abrupt_stop_from30", [("hold", 4.0), ("acc", -4.5, 0.0), ("hold", 6.0)], ev0=30 * MPH, lv0=30 * MPH, from_stop=False),
  S("cutin_12mph_then_stop", [("hold", 4.5), ("acc", -1.5, 0.0), ("hold", 6.0)], ev0=12 * MPH, lv0=12 * MPH,
    cut=dict(t=3.0, gap=7.0, v=10 * MPH), from_stop=False),
  S("cutin_crawl_close", [("hold", 3.0), ("acc", -2.0, 0.0), ("hold", 6.0)], ev0=5 * MPH, lv0=5 * MPH,
    cut=dict(t=2.0, gap=4.0, v=4 * MPH), from_stop=False),
  S("city_lights_40mph", [("hold", 4.0), ("acc", -1.5, 0.0), ("hold", 12.0), ("acc", 1.5, 35 * MPH), ("hold", 8.0),
                          ("acc", -2.5, 0.0), ("hold", 8.0), ("acc", 1.0, 45 * MPH), ("hold", 6.0)],
    ev0=40 * MPH, lv0=40 * MPH, v_cruise=45 * MPH, from_stop=False),
  S("city_speed_changes_25_45", [("hold", 3.0), ("acc", 0.8, 45 * MPH), ("hold", 4.0), ("acc", -1.0, 30 * MPH), ("hold", 3.0),
                                 ("acc", 0.6, 40 * MPH), ("hold", 3.0), ("acc", -1.5, 25 * MPH), ("hold", 5.0)],
    ev0=25 * MPH, lv0=25 * MPH, v_cruise=50 * MPH, from_stop=False),
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


def run(personality, sc, cand):
  exp = False
  cfg = cand.get("sng")
  sng = SNG.StopAndGo(SNG.SngConfig(**cfg)) if cfg is not None else None
  use_sl = bool(sng is not None and sng.config.safe_launch and SafeLaunch is not None)
  sl = SafeLaunch() if use_sl else None
  use_rel = bool(cand.get("release", False) and release_prev_accel is not None)
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
  # warm start at steady speed
  if ev > 0.1:
    for _ in range(40):
      mpc.set_weights(True, personality=personality, experimental_mode=exp)
      mpc.set_cur_state(ev, 0.0)
      mpc.update(H._radar(gap0, lv, 0.0), personality=personality, experimental_mode=exp)
  cut_done = False
  solve_fail = 0
  rng = np.random.default_rng(1234)
  t = 0.0; k = 0
  min_gap = 1e9; min_gap_stop_rel = 1e9; peak_a = 0.0; peak_d = 0.0; peak_j = 0.0; jerk_sq = 0.0; n_j = 0
  ea_hist = []; v_hist = []; collided = False; v_peak = 0.0
  stop_decels = []; v_prev_step = ev; ea_prev_step = 0.0
  moves = []  # (lead_start_t, ego_move_t)
  lead_was_stopped = lv < 0.05; t_lead_start = None; ego_stopped = ev < 0.05
  reversals = 0; last_sign = 0
  w_max = 0.0; w_time = 0.0; trace = []
  t_end_lead = None
  t_max = 240.0
  while t < t_max:
    # lead
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
    # lead-start bookkeeping (from a stop)
    if lv < 0.05:
      lead_was_stopped = True
    elif lead_was_stopped and lv > 0.05:
      lead_was_stopped = False
      if ev < 0.05:
        t_lead_start = t
    if t_lead_start is not None and ev > 0.1:
      moves.append(round(t - t_lead_start, 3)); t_lead_start = None
    # planner 20 Hz
    if k % 5 == 0:
      if abs(la_f) < 0.5:
        tau_f = 1.5
      else:
        tau_f += (0.0 - tau_f) * (0.05 / (0.45 + 0.05))
      if sc["noise"]:
        rs = H._radar(gap + rng.normal(0, 0.15), max(lv + rng.normal(0, 0.15), 0.0), la_f + rng.normal(0, 0.3), tau=tau_f)
      else:
        rs = H._radar(gap, lv, la_f, tau=tau_f)
      rs.leadOne.vRel = float(lv - ev)
      standstill = ev < 1e-3
      cap = float(LP.get_max_accel(ev, personality == P.aggressive, personality == P.relaxed))
      vdf.x = max(0.0, vdf.update(ev))
      pac = not standstill
      fcw_prev = mpc.crash_cnt > 2 and not standstill
      if sl is not None:
        if sng.config.safe_launch_brake_free:
          pac = sl.prev_accel_constraint(pac, ev, rs.leadOne, desired_follow_gap(ev, lv, personality, exp), fcw_prev)
        else:
          pac = pac and not launch_gate_ok(ev, rs.leadOne, desired_follow_gap(ev, lv, personality, exp), fcw_prev)
      jf = None; tf = None; dcost = None; mpc.stop_extra_predicted = 0.0
      if sng is not None:
        o = sng.update(ev, rs.leadOne, LM.get_jerk_factor(personality, exp), cap, smooth_accel_cap(ev, "relaxed"),
                       desired_follow_gap(ev, lv, personality, exp), out_a, fcw_prev, 0.05,
                       base_t_follow=LM.get_T_FOLLOW(personality, exp, vdf.x),
                       plan_a_min=float(np.min(mpc.a_solution[:SNG_PLAN_N])))
        jf, cap, tf, dcost = o.jerk_factor, o.cap, o.t_follow, o.danger_cost
        mpc.stop_extra_predicted = o.stop_extra_predicted
        mpc.lead_danger_factor = o.danger_factor if o.danger_factor is not None else LM.LEAD_DANGER_FACTOR
        if o.brake_free:
          pac = False
        w_max = max(w_max, o.weight); w_time += 0.05 * o.weight
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
      ss = should_stop(ev, a_mpc) or should_stop(ev, a_cruise)
      a_t = float(np.clip(min(a_mpc, a_cruise), ACCEL_MIN, cap))
      a_t = float(LP.limit_accel_rise(a_t, a_prev, ev, 0.05))
      out_a = a_t; plan_a = a_t
      vdf.x = vdf.x + 0.05 * (out_a + a_prev) / 2.0
    # controls 100 Hz
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
    delay.append(sent); d = delay.pop(0)
    ea = ea + (d - ea) * (1 - math.exp(-dt_c / 0.25))
    ea_eff = 0.0 if (ev <= 0.0 and ea < 0.0) else ea
    ev = max(0.0, ev + ea_eff * dt_c)
    ex += ev * dt_c
    gap = lx - ex
    # metrics
    if not (cut and not cut_done):
      min_gap = min(min_gap, gap)
      if lv < 0.05 and ev < 0.05:
        min_gap_stop_rel = min(min_gap_stop_rel, gap - g_stop)
    if gap <= 0.0: collided = True
    v_peak = max(v_peak, ev)
    peak_a = max(peak_a, ea_eff); peak_d = min(peak_d, ea_eff)
    ea_hist.append(ea_eff); v_hist.append(ev)
    if len(v_hist) > 6: v_hist.pop(0)
    v_hist_min = min(v_hist)
    if ev <= 0.0 and v_prev_step > 0.0:
      stop_decels.append(round(ea_prev_step, 3))
    v_prev_step = ev; ea_prev_step = ea_eff
    if len(ea_hist) > 5:
      j = (ea_hist[-1] - ea_hist[-6]) / 0.05
      ea_hist.pop(0)
      if ev > 0.1 and v_hist_min > 0.1:
        peak_j = max(peak_j, abs(j)); jerk_sq += j * j; n_j += 1
    s = 1 if ea_eff > 0.15 else (-1 if ea_eff < -0.15 else 0)
    if s != 0 and last_sign != 0 and s != last_sign:
      reversals += 1
    if s != 0: last_sign = s
    if k % 20 == 0:
      trace.append([round(t, 2), round(gap, 2), round(lv / MPH, 2), round(ev / MPH, 2), round(plan_a, 3), round(ea_eff, 3)])
    t += dt_c; k += 1
  return dict(collision=collided, min_gap_m=round(min_gap, 2), stop_gap_m=round(g_stop, 2),
              min_gap_minus_stop_gap_m=round(min_gap - g_stop, 2),
              standstill_gap_minus_stop_gap_m=None if min_gap_stop_rel > 1e8 else round(min_gap_stop_rel, 2),
              final_gap_m=round(lx - ex, 2), peak_accel=round(peak_a, 3), peak_decel=round(peak_d, 3),
              peak_jerk=round(peak_j, 2), rms_jerk=round(math.sqrt(jerk_sq / max(n_j, 1)), 3),
              accel_reversals=reversals, stop_decels=stop_decels,
              stop_decel_worst=round(min(stop_decels), 3) if stop_decels else None, t_move=moves, t_move_mean=round(float(np.mean(moves)), 3) if moves else None,
              t_move_max=round(float(max(moves)), 3) if moves else None,
              v_ego_peak_mph=round(v_peak / MPH, 2), sng_weight_max=round(w_max, 2), sng_time_s=round(w_time, 1),
              duration_s=round(t, 1), solve_fail=solve_fail, trace=trace)


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
    print(f"{CAND.get('name','?'):10s} {PNAME[p]:10s} {sc['name']:26s} coll={int(r['collision'])} "
          f"min_gap={r['min_gap_m']:6.2f} (stop {r['stop_gap_m']}) sgap={r['standstill_gap_minus_stop_gap_m']} "
          f"a+={r['peak_accel']:.2f} a-={r['peak_decel']:.2f} jpk={r['peak_jerk']:.2f} jrms={r['rms_jerk']:.3f} "
          f"fail={r['solve_fail']} rev={r['accel_reversals']} stopdec={r['stop_decel_worst']} tmove={r['t_move_mean']}/{r['t_move_max']} w={r['sng_weight_max']}", flush=True)
OUT.write_text(json.dumps(res, indent=1))
