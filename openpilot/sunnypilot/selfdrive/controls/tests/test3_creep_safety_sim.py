#!/usr/bin/env python3
"""Stop-and-go creep safety sim, run against one tip's real code.

usage: test3_creep_safety_sim.py TIP_ROOT BUILD_DIR OUT_JSON [HELPER_REPO] [personalities] [case-substrings]

Regression for the t305u/t308u near-collision on d79496cac9: both cars
stopped, the lead creeps 1.5-3 m (2-5 mph) and stops, and ego must not
run up on it. HELPER_REPO defaults to TIP_ROOT.

Uses the tip's own LongitudinalMpc, planner add-ons (LeadLaunch,
FinalStopTaper, LeadSlow when present), get_accel_from_plan,
should_stop, cruise accel, caps, and real LongControl. Toyota TSS2
carcontroller long path emulated line-for-line (4.0 pre-PID rate limit,
TSS2 PID with freeze when not pid, unwind, a_ego_future), plus the
tip's smooth_stop_hold when present. Plant: 0.05 s delay + tau 0.25.
Official tip: candidates = min(MPC, cruise) (the model e2e candidate
is not simulated). Not road-tested.
"""
import importlib.util, inspect, json, sys, types
from pathlib import Path
from types import SimpleNamespace

TIP = Path(sys.argv[1]).resolve()
BUILD = Path(sys.argv[2])
OUT = Path(sys.argv[3])
HELPER_REPO = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else TIP
MPH = 0.44704
FT = 0.3048

sys.path[:0] = [str(TIP), str(TIP / "opendbc_repo"), str(HELPER_REPO / "msgq_repo"), "/tmp/test2-pydeps"]
helper = HELPER_REPO / "openpilot/sunnypilot/selfdrive/controls/tests/test3_c3x_equiv_sim.py"
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

def try_import(mod, name):
  try:
    m = __import__(mod, fromlist=[name]); return getattr(m, name)
  except Exception:
    return None

LeadLaunch = try_import("openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_launch", "LeadLaunch")
desired_follow_gap = try_import("openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_launch", "desired_follow_gap")
FinalStopTaper = try_import("openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_final_stop", "FinalStopTaper")
LeadSlow = try_import("openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_lead_slow", "LeadSlow")
smooth_pcm_accel = try_import("opendbc.sunnypilot.car.toyota.smooth_stop_hold", "smooth_pcm_accel")
fade_stopping_integral = try_import("opendbc.sunnypilot.car.toyota.smooth_stop_hold", "fade_stopping_integral")
ChillAggressiveLaunch = try_import("openpilot.sunnypilot.selfdrive.car.chill_aggressive", "ChillAggressiveLaunch")
P = log.LongitudinalPersonality
LongCtrlState = car.CarControl.Actuators.LongControlState
CTRL_T = ModelConstants.T_IDXS[:CONTROL_N]
ACCEL_MIN, ACCEL_MAX = -3.5, 2.0

def has_kw(fn, kw):
  try: return kw in inspect.signature(fn).parameters
  except Exception: return False

def lead_profile(a_up, v_peak, a_dn):
  """Lead starts at t0, accel a_up to v_peak, then a_dn to 0."""
  return a_up, v_peak, a_dn

def run(personality, a_up, v_peak, a_dn, t_end=22.0):
  exp = False
  mpc = LM.LongitudinalMpc()
  extra = LM.get_lead_stop_extra(personality, exp) if hasattr(LM, "get_lead_stop_extra") else 0.0
  gap0 = LM.STOP_DISTANCE + extra
  ll = LeadLaunch() if LeadLaunch else None
  fs = FinalStopTaper() if FinalStopTaper else None
  ls = LeadSlow() if LeadSlow else None
  cal = ChillAggressiveLaunch() if (ChillAggressiveLaunch and hasattr(LP, "get_max_accel")) else None
  CP = SimpleNamespace(stopAccel=-2.0, vEgoStopping=0.25, vEgoStarting=0.25,
                       longitudinalTuning=SimpleNamespace(kiBP=[0.], kiV=[0.]))
  CP_SP = SimpleNamespace(enableGasInterceptor=False)
  lc = LongControl(CP, CP_SP)
  pid = PIDController(0.0, ([2., 5.], [0.5, 0.25]), k_f=1.0, pos_limit=2.0, neg_limit=-3.5, rate=1/0.03)
  aego_f = FirstOrderFilter(0.0, 0.25, 0.03)
  vdf = FirstOrderFilter(0.0, 2.0, 0.05)
  v_cruise = 40 * MPH
  dt_c = 0.01
  # state
  ex, ev, ea = 0.0, 0.0, 0.0
  lx, lv, la_true, la_f = gap0, 0.0, 0.0, 0.0
  plan_a, ss, a_cruise = 0.0, True, 0.0
  out_a = 0.0
  prev_accel = 0.0; sent = 0.0
  delay = [0.0] * 5
  t0 = 3.0
  phase = "wait"; t_lead_brake = None; t_lead_stop = None
  t_ego_move = None; t_first_brake_cmd = None; t_first_brake_plant = None
  min_gap = 1e9; peak_dec = 0.0; v_ego_peak = 0.0; collided = False; t_coll = None
  lead_moved = 0.0; trace = []; a_raw_dbg = 0.0
  for k in range(int(t_end / dt_c)):
    t = k * dt_c
    # lead
    if t >= t0 and phase == "wait": phase = "up"
    if phase == "up":
      la_true = a_up
      if lv >= v_peak: phase = "down"; t_lead_brake = t
    if phase == "down":
      la_true = a_dn
      if lv <= 0.0: phase = "stopped"; t_lead_stop = t
    if phase in ("wait", "stopped"): la_true = 0.0
    lv = max(0.0, lv + la_true * dt_c)
    if phase == "stopped": lv = 0.0
    lx += lv * dt_c
    la_f += (la_true - la_f) * (1 - np.exp(-dt_c / 0.3))  # radar aLeadK lag
    gap = lx - ex
    # planner 20 Hz
    if k % 5 == 0:
      rs = H._radar(gap, lv, la_f)
      rs.leadOne.vRel = float(lv - ev)
      standstill = ev < 1e-3
      if cal is not None:
        boost = cal.update(True, exp, personality, ev, v_cruise, False)
        if hasattr(LP, "smooth_accel_cap"):  # t314u: Agg curve is personality-based
          cap = float(LP.get_max_accel(ev, personality == P.aggressive, personality == P.relaxed))
        else:
          cap = float(LP.get_max_accel(ev, boost, personality == P.relaxed))
      else:
        cap = ACCEL_MAX
      vdf.x = max(0.0, vdf.update(ev))
      if has_kw(mpc.set_weights, "experimental_mode"):
        mpc.set_weights(not standstill, personality=personality, experimental_mode=exp)
      else:
        mpc.set_weights(not standstill, personality=personality)
      mpc.set_cur_state(vdf.x, out_a)
      if has_kw(mpc.update, "experimental_mode"):
        mpc.update(rs, personality=personality, experimental_mode=exp)
      else:
        mpc.update(rs, personality=personality)
      vt = np.interp(CTRL_T, LM.T_IDXS, mpc.v_solution)
      at = np.interp(CTRL_T, LM.T_IDXS, mpc.a_solution)
      a_prev = out_a
      a_mpc = get_accel_from_plan(vt, at, CTRL_T, action_t=0.05 + 0.05)
      ss_mpc = should_stop(ev, a_mpc)
      if has_kw(LP.get_cruise_accel, "cap"):
        a_cruise = LP.get_cruise_accel(v_cruise, ev, a_cruise, 0.05, cap=cap)
      else:
        a_cruise = LP.get_cruise_accel(v_cruise, ev, a_cruise, 0.05)
      ss_cr = should_stop(ev, a_cruise)
      a_t = min(a_mpc, a_cruise); ss = ss_mpc or ss_cr
      a_t = float(np.clip(a_t, ACCEL_MIN, cap))
      lead = rs.leadOne
      fcw = mpc.crash_cnt > 2 and not standstill
      if ll is not None:
        kw = dict(cap=cap, jerk_up=0.40, v_ego_stopping=0.25)
        if has_kw(ll.apply, "desired_gap"):
          kw["desired_gap"] = desired_follow_gap(ev, lv, personality, exp)
        a_t, ss = ll.apply(a_t, ss, ev, v_cruise, lead, fcw, 0.05, **kw)
        a_t = float(np.clip(a_t, ACCEL_MIN, cap))
      if ls is not None:
        a_t = ls.apply(a_t, ev, lead, personality, fcw, 0.05,
                       desired_gap=desired_follow_gap(ev, lv, personality, exp), a_prev=a_prev,
                       experimental_mode=exp)
      if fs is not None:
        kw = dict(should_stop=bool(ss), a_prev=a_prev, min_gap=3.5)
        if has_kw(fs.apply, "personality"):
          kw.update(personality=personality, experimental_mode=exp,
                    desired_gap=desired_follow_gap(ev, lv, personality, exp))
        a_t = fs.apply(a_t, ev, lead, fcw, 0.05, **kw)
        a_t = float(np.clip(a_t, ACCEL_MIN, cap))
      if hasattr(LP, "limit_accel_rise"):
        a_t = float(LP.limit_accel_rise(a_t, a_prev, ev, 0.05))
      out_a = a_t; plan_a = a_t; a_raw_dbg = float(min(a_mpc, a_cruise))
      vdf.x = vdf.x + 0.05 * (out_a + a_prev) / 2.0
    # controls 100 Hz
    CS = SimpleNamespace(vEgo=ev, aEgo=ea, brakePressed=False, standstill=ev < 1e-3,
                         cruiseState=SimpleNamespace(standstill=False))
    act = float(lc.update(True, CS, plan_a, ss, (ACCEL_MIN, ACCEL_MAX)))
    stopping = lc.long_control_state == LongCtrlState.stopping
    # Toyota carcontroller (every 3 frames)
    if k % 3 == 0:
      pcm = rate_limit(act, prev_accel, -4.0 * 0.03, 4.0 * 0.03)
      prev_accel = pcm
      prev_ae = aego_f.x; aego_f.update(ea)
      j_ego = (aego_f.x - prev_ae) / 0.03
      fut = float(np.interp(ev, [2., 5.], [0.25, 0.5]))
      a_fut = ea + j_ego * fut
      pid.i -= 0.03 * 0.03 * float(np.sign(pid.i))
      pcm = pid.update(pcm - a_fut, speed=ev, feedforward=pcm,
                       freeze_integrator=lc.long_control_state != LongCtrlState.pid)
      pcm = float(np.clip(pcm, -3.5, 2.0))
      if smooth_pcm_accel is not None:
        pcm = smooth_pcm_accel(pcm, act, sent, 0.03, stopping, ev, True)
        if stopping: fade_stopping_integral(pid, 0.03)
      sent = pcm
    delay.append(sent); d = delay.pop(0)
    ea = ea + (d - ea) * (1 - np.exp(-dt_c / 0.25))
    if ev <= 0.0 and ea < 0.0:
      ea_eff = 0.0
    else:
      ea_eff = ea
    ev = max(0.0, ev + ea_eff * dt_c)
    ex += ev * dt_c
    gap = lx - ex
    if t >= t0:
      if t_ego_move is None and ev > 0.1: t_ego_move = t
      if t_lead_brake is not None and t >= t_lead_brake:
        if t_first_brake_cmd is None and sent <= -0.5: t_first_brake_cmd = t
        if t_first_brake_plant is None and ea <= -0.5: t_first_brake_plant = t
      min_gap = min(min_gap, gap); peak_dec = min(peak_dec, ea); v_ego_peak = max(v_ego_peak, ev)
      if gap <= 0.0 and not collided: collided = True; t_coll = t
    if k % 10 == 0 and t >= t0:
      trace.append([round(t, 2), round(gap, 2), round(lv, 2), round(ev, 2), round(plan_a, 3),
                    round(act, 3), round(sent, 3), round(ea, 3), int(stopping), round(a_raw_dbg, 3)])
    if phase == "stopped" and t_lead_stop is not None and t > t_lead_stop + 6.0 and ev < 0.01:
      break
  lead_moved = lx - gap0
  rel = lambda x: None if (x is None or t_lead_brake is None) else round(x - t_lead_brake, 2)
  return dict(gap0_ft=round(gap0 / FT, 2), lead_moved_m=round(lead_moved, 2),
              min_gap_m=round(min_gap, 2), min_gap_ft=round(min_gap / FT, 2),
              final_gap_ft=round(gap / FT, 2), collision=collided,
              t_ego_move_after_lead=None if t_ego_move is None else round(t_ego_move - t0, 2),
              brake_cmd_delay_s=rel(t_first_brake_cmd), brake_plant_delay_s=rel(t_first_brake_plant),
              peak_decel=round(peak_dec, 3), v_ego_peak_mph=round(v_ego_peak / MPH, 2),
              trace=trace, lead_peak_mph=round(v_peak / MPH, 2))

def v_peak_for_creep(D, a_up, a_dn):
  return float(np.sqrt(2 * D / (1 / a_up + 1 / abs(a_dn))))

cases = []
for a_up in (0.5, 1.0, 1.5):
  for vp_mph in (3, 6, 10):
    for a_dn in (-1.5, -3.0):
      cases.append((f"up{a_up}_to{vp_mph}mph_dn{a_dn}", a_up, vp_mph * MPH, a_dn))
for D in (1.0, 1.5, 2.0, 3.0):
  for a_up, a_dn in ((1.0, -1.5), (1.0, -3.0), (0.5, -1.5)):
    vp = v_peak_for_creep(D, a_up, a_dn)
    cases.append((f"creep{D}m_up{a_up}_dn{a_dn}_vp{vp/MPH:.1f}mph", a_up, vp, a_dn))

res = {"tip": TIP.name, "has": dict(LeadLaunch=bool(LeadLaunch), FinalStopTaper=bool(FinalStopTaper),
       LeadSlow=bool(LeadSlow), smooth_pcm=bool(smooth_pcm_accel)), "cases": {}}
pers = [("aggressive", P.aggressive), ("standard", P.standard), ("relaxed", P.relaxed)]
only = set(sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] else None
subs = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else None
if subs:
  cases = [c for c in cases if any(s in c[0] for s in subs)]
for pname, p in pers:
  if only and pname not in only: continue
  res["cases"][pname] = {}
  for name, a_up, vp, a_dn in cases:
    r = run(p, a_up, vp, a_dn)
    res["cases"][pname][name] = r
    print(f"{TIP.name} {pname:10s} {name:34s} moved={r['lead_moved_m']:5.2f}m min_gap={r['min_gap_m']:5.2f}m "
          f"coll={int(r['collision'])} brake_cmd_delay={r['brake_cmd_delay_s']} peak={r['peak_decel']} "
          f"vpk={r['v_ego_peak_mph']}", flush=True)
OUT.write_text(json.dumps(res, indent=1))
