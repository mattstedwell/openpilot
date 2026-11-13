#!/usr/bin/env python3
"""devc4-test3 closed-loop sims: follow gaps, launch-follow, lead braking,
cut-in inside the floor, and last-5-mph / hold-step before vs after.

usage: test3_floor_follow_sim.py REPO [BUILD_DIR] [out.json]
Nothing road-tested.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from openpilot.sunnypilot.selfdrive.controls.tests.test3_c3x_equiv_sim import (
  _setup, _radar, _plan, BUILD, ROOT,
)

MPH = 0.44704
FT = 0.3048


def _hooks(a_t, ss, ev, v_cruise, lead, dt, cap, personality, exp,
           use_launch, use_taper, use_floor, use_boost_gate,
           ls, fs, launch_boost, get_max_accel, is_relaxed, is_agg,
           LeadLaunch, FinalStopTaper, apply_min_gap_floor,
           desired_bumper_gap, get_min_gap_floor, a_prev, fcw=False):
  """Apply the planner-side hooks in the same order as the planner."""
  import numpy as np
  d_rel = getattr(lead, "dRel", float("nan"))
  v_lead = getattr(lead, "vLead", float("nan"))
  desired = desired_bumper_gap(ev, v_lead, personality, exp)
  floor = get_min_gap_floor(personality, exp)
  boost = launch_boost
  if use_boost_gate and np.isfinite(d_rel) and d_rel < desired:
    boost = False
  cap = float(get_max_accel(ev, boost if is_agg else False, is_relaxed))
  if a_t > 0:
    a_t = min(a_t, cap)
  if use_launch and ls is not None:
    a_t, ss = ls.apply(
      a_t, ss, ev, v_cruise, lead, fcw, dt, cap=cap,
      desired_gap=desired if use_launch else None,
      min_gap_floor=floor if use_floor else None)
    if a_t > 0:
      a_t = min(a_t, cap)
  if use_taper and fs is not None:
    a_t = fs.apply(a_t, ev, lead, fcw, dt, should_stop=bool(ss),
                   a_prev=a_prev, min_gap=floor if use_floor else 3.5)
  if use_floor:
    a_t, ss, _ = apply_min_gap_floor(
      a_t, ss, d_rel, ev, v_lead, a_prev, floor, dt, fcw=fcw)
  return a_t, ss, cap, desired, floor


def run_follow(personality, mode, target_mph, kw, kind="launch"):
  """kind='launch': stop behind lead, lead accel 1.0 to target and holds.
  kind='steady': start already at speed at the desired gap, 15 s settle.
  kind='brake': follow at target, then lead brakes at -1.5 to a stop.
  kind='cutin': follow at target at desired, then lead appears at 15 ft.
  """
  import numpy as np
  get_max_accel = kw["get_max_accel"]
  LeadLaunch = kw["LeadLaunch"]
  FinalStopTaper = kw["FinalStopTaper"]
  apply_min_gap_floor = kw["apply_min_gap_floor"]
  desired_bumper_gap = kw["desired_bumper_gap"]
  get_min_gap_floor = kw["get_min_gap_floor"]
  ChillAggressiveLaunch = kw["ChillAggressiveLaunch"]
  P = kw["P"]
  get_lead_stop_extra = kw["get_lead_stop_extra"]
  STOP_DISTANCE = kw["STOP_DISTANCE"]
  LongitudinalMpc = kw["LongitudinalMpc"]
  get_T_FOLLOW = kw["get_T_FOLLOW"]
  stopping_accel_step = kw["stopping_accel_step"]

  run_follow._cut = False
  after = mode == "after"
  use_launch = after or mode == "legacy"
  use_taper = after
  use_floor = after
  use_boost_gate = after
  use_smooth_hold = after

  dt_mdl, dt_ctrl, tau = 0.05, 0.01, 0.25
  is_relaxed = personality == P.relaxed
  is_agg = personality == P.aggressive
  exp = False
  floor0 = STOP_DISTANCE + get_lead_stop_extra(personality, exp)
  T = get_T_FOLLOW(personality, exp, 0.0)
  v_tgt = target_mph * MPH
  v_cruise = 25.0

  mpc = LongitudinalMpc()
  ls = LeadLaunch() if use_launch else None
  fs = FinalStopTaper() if use_taper else None
  boost_sm = ChillAggressiveLaunch()

  if kind == "steady":
    ev = v_tgt
    desired0 = desired_bumper_gap(ev, ev, personality, exp)
    ex, ea = 0.0, 0.0
    lx, lv = desired0, v_tgt
    plan_a, v_des, last, state, ss = 0.0, ev, 0.0, "pid", False
    sim_t = 20.0
  elif kind == "brake":
    ev = v_tgt
    desired0 = desired_bumper_gap(ev, ev, personality, exp)
    ex, ea = 0.0, 0.0
    lx, lv = desired0, v_tgt
    plan_a, v_des, last, state, ss = 0.0, ev, 0.0, "pid", False
    sim_t = 25.0
  elif kind == "cutin":
    ev = v_tgt
    desired0 = desired_bumper_gap(ev, ev, personality, exp)
    ex, ea = 0.0, 0.0
    lx, lv = desired0, v_tgt
    plan_a, v_des, last, state, ss = 0.0, ev, 0.0, "pid", False
    sim_t = 15.0
  else:  # launch
    ev = 0.0
    ex, ea = 0.0, 0.0
    lx, lv = floor0, 0.0
    plan_a, v_des, last, state, ss = -0.5, 0.0, -0.5, "stopping", True
    sim_t = 45.0 if target_mph >= 40 else 40.0

  a_cruise_st = plan_a
  min_gap = 1e9
  min_25_30 = 1e9
  at_speed = {}
  traces = []
  below_floor = False
  t_below = None
  cutin_at = 3.0
  brake_at = 3.0
  launch_on = False
  peak_a_cmd = -1e9
  peak_j_cmd = 0.0
  last5 = {"peak_a_5mph": 0.0, "peak_j_5mph": 0.0, "peak_j_5mph_ex_art": 0.0,
           "peak_lc_j_5mph": 0.0, "peak_lc_j_5mph_ex_art": 0.0,
           "a_plan_at_0": None, "a_lc_at_0": None,
           "hold_step_0_2s": None, "hold_rate": None,
           "standstill_step_0_2s": None, "standstill_rate": None,
           "entered_stopping": None, "entered_standstill": None,
           "a_at_8mph": None, "a_at_5mph": None}
  last5_active = False
  hold_t0 = None
  hold_a0 = None
  ss_t0 = None
  ss_a0 = None
  prev_lc = last
  prev_pa = plan_a
  in_5 = False

  for k in range(int(sim_t / dt_ctrl)):
    t = k * dt_ctrl
    # lead motion
    la = 0.0
    if kind == "launch":
      la = 0.0 if lv >= v_tgt else 1.0
      if lv < v_tgt and lv + la * dt_ctrl > v_tgt:
        la = (v_tgt - lv) / dt_ctrl
      lv = min(max(lv + la * dt_ctrl, 0.0), v_tgt)
    elif kind == "brake":
      if t >= brake_at and lv > 0:
        la = -1.5
        if lv + la * dt_ctrl < 0:
          la = -lv / dt_ctrl
        lv = max(lv + la * dt_ctrl, 0.0)
      else:
        lv = v_tgt if t < brake_at else lv
    elif kind == "cutin":
      lv = v_tgt
      if t >= cutin_at and not getattr(run_follow, "_cut", False):
        lx = ex + 15 * FT  # appear at 15 ft, once
        run_follow._cut = True
    else:
      lv = v_tgt
    lx += lv * dt_ctrl

    if k % 5 == 0:
      d_rel = lx - ex
      rs = _radar(d_rel, lv, la, present=True)
      rs.leadOne.vRel = float(lv - ev)
      launch_boost = boost_sm.update(True, False, personality, ev, v_cruise, False)
      cap = float(get_max_accel(ev, launch_boost if is_agg else False, is_relaxed))
      a_t, ss, a_cruise_st = _plan(
        mpc, ev, plan_a, v_des, rs, v_cruise, personality, False, cap,
        a_cruise_prev=a_cruise_st)
      a_t, ss, cap, desired, floor = _hooks(
        a_t, ss, ev, v_cruise, rs.leadOne, dt_mdl, cap, personality, False,
        use_launch, use_taper, use_floor, use_boost_gate,
        ls, fs, launch_boost, get_max_accel, is_relaxed, is_agg,
        LeadLaunch, FinalStopTaper, apply_min_gap_floor,
        desired_bumper_gap, get_min_gap_floor, plan_a)
      plan_a = float(np.clip(a_t, -3.5, 2.5))
      v_des = 0.5 * max(0.0, v_des + dt_mdl * plan_a) + 0.5 * ev
      if ev < 0.25:
        ss = plan_a < 0.1
      launch_on = bool(ls and (ls.launch.active or ls.launch.go_count > 0))

    # longcontrol
    if state == "stopping" and (not ss):
      state = "pid"
    elif state == "pid" and ss and ev < 0.25:
      state = "stopping"
      if last5_active and last5["entered_stopping"] is None:
        last5["entered_stopping"] = {"t": t, "v_mph": ev / MPH, "a": last}
        hold_t0, hold_a0 = t, last
    if state == "stopping":
      if use_smooth_hold:
        out = stopping_accel_step(last, ev <= 0.01, dt_ctrl, ev, stop_accel=-2.0)
        out = max(out, -2.0)
      else:
        out = last
        if out > -2.0:
          out = min(out, 0.0) - 1.0 * dt_ctrl
        out = max(out, -2.0)
    else:
      out = float(ea + (plan_a - ea) * (1.0 - np.exp(-dt_ctrl / tau)))
    # last-5-mph and hold metrics
    ev_mph = ev / MPH if ev else 0.0
    j_pa = abs(plan_a - prev_pa) / dt_mdl if k % 5 == 0 else 0.0
    j_lc = abs(out - prev_lc) / dt_ctrl
    if ev_mph <= 8.05 and ev_mph > 0.05 and (kind == "brake" or ss or plan_a < -0.05):
      last5_active = True
      if last5["a_at_8mph"] is None and ev_mph <= 8.05:
        last5["a_at_8mph"] = round(plan_a, 3)
      if last5["a_at_5mph"] is None and ev_mph <= 5.05:
        last5["a_at_5mph"] = round(plan_a, 3)
      if ev_mph <= 5.05:
        last5["peak_a_5mph"] = min(last5["peak_a_5mph"], plan_a)
        last5["peak_j_5mph"] = max(last5["peak_j_5mph"], j_pa)
        if j_pa < 5.0:  # exclude one-frame artifacts
          last5["peak_j_5mph_ex_art"] = max(last5["peak_j_5mph_ex_art"], j_pa)
        last5["peak_lc_j_5mph"] = max(last5["peak_lc_j_5mph"], j_lc)
        if j_lc < 5.0:
          last5["peak_lc_j_5mph_ex_art"] = max(last5["peak_lc_j_5mph_ex_art"], j_lc)
    if ev <= 0.01 and last5["a_lc_at_0"] is None and last5_active:
      last5["a_plan_at_0"] = round(plan_a, 3)
      last5["a_lc_at_0"] = round(out, 3)
      if ev <= 0.01 and ss_t0 is None:
        ss_t0, ss_a0 = t, out
        last5["entered_standstill"] = {"t": round(t, 2), "a": round(out, 3)}
    if hold_t0 is not None:
      if t - hold_t0 >= 0.2 and last5["hold_step_0_2s"] is None:
        last5["hold_step_0_2s"] = round(out - hold_a0, 3)
        last5["hold_rate"] = round((out - hold_a0) / max(t - hold_t0, 1e-6), 3)
    if ss_t0 is not None:
      if t - ss_t0 >= 0.2 and last5["standstill_step_0_2s"] is None:
        last5["standstill_step_0_2s"] = round(out - ss_a0, 3)
        last5["standstill_rate"] = round((out - ss_a0) / max(t - ss_t0, 1e-6), 3)
    prev_pa = plan_a
    last = out
    prev_lc = out
    ea = out
    ev = max(0.0, ev + ea * dt_ctrl)
    if ev <= 0 and ea < 0:
      ev = 0.0
    ex += ev * dt_ctrl
    gap = lx - ex
    min_gap = min(min_gap, gap)
    ev_mph = ev / MPH
    if 25 <= ev_mph <= 32:
      min_25_30 = min(min_25_30, gap)
    if gap < floor0 - 0.02:
      below_floor = True
      if t_below is None:
        t_below = t
    tg = gap / ev if ev > 0.3 else None
    T_now = get_T_FOLLOW(personality, False, ev)
    desired = T_now * ev + STOP_DISTANCE  # extras out at speed
    t_eq = (gap - STOP_DISTANCE) / ev if ev > 0.3 else None
    for mph in (5, 10, 15, 20, 25, 30, 35, 40, 45):
      if mph not in at_speed and ev >= mph * MPH and kind == "launch":
        at_speed[mph] = {
          "t": round(t, 2), "gap_ft": round(gap / FT, 1),
          "time_gap_s": None if tg is None else round(tg, 2),
          "T_eq": None if t_eq is None else round(t_eq, 2),
          "desired_ft": round(desired / FT, 1),
          "launch_active": launch_on,
          "below_floor": gap < floor0 - 0.02,
        }
    if k % 50 == 0:
      traces.append({
        "t": round(t, 2), "ev_mph": round(ev_mph, 2),
        "gap_ft": round(gap / FT, 1),
        "tg": None if tg is None else round(tg, 2),
        "T_eq": None if t_eq is None else round(t_eq, 2),
        "des": round(desired / FT, 1),
        "lnch": launch_on, "pa": round(plan_a, 3),
        "lc": round(out, 3), "ss": ss, "st": state,
      })
    peak_a_cmd = max(peak_a_cmd, plan_a)
    if traces and k % 5 == 0:
      pass

  tail = [x for x in traces if x["t"] >= sim_t - 3]
  def avg(key):
    xs = [x[key] for x in tail if x.get(key) is not None]
    return None if not xs else round(sum(xs) / len(xs), 2)

  return {
    "T": T,
    "floor_ft": round(floor0 / FT, 2),
    "kind": kind,
    "mode": mode,
    "target_mph": target_mph,
    "at_speed": {str(k): at_speed[k] for k in sorted(at_speed)},
    "min_gap_ft": round(min_gap / FT, 1),
    "min_25_30_ft": None if min_25_30 > 1e8 else round(min_25_30 / FT, 1),
    "below_floor": below_floor,
    "t_below": t_below,
    "steady_gap_ft": avg("gap_ft"),
    "steady_tg": avg("tg"),
    "steady_T_eq": avg("T_eq"),
    "end_gap_ft": traces[-1]["gap_ft"] if traces else None,
    "end_v_mph": traces[-1]["ev_mph"] if traces else None,
    "end_pa": traces[-1]["pa"] if traces else None,
    "last5": last5,
    "launch_on_at_end": traces[-1]["lnch"] if traces else None,
  }


def main():
  from openpilot.cereal import log
  from opendbc.car.interfaces import ACCEL_MAX
  _setup(BUILD, ROOT)
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
    LongitudinalMpc, get_lead_stop_extra, get_T_FOLLOW, STOP_DISTANCE,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel
  from openpilot.sunnypilot.selfdrive.car.chill_aggressive import ChillAggressiveLaunch
  from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_launch import LeadLaunch
  from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_final_stop import (
    FinalStopTaper, stopping_accel_step,
  )
  from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_min_gap import (
    apply_min_gap_floor, desired_bumper_gap, get_min_gap_floor,
  )
  P = log.LongitudinalPersonality
  kw = dict(
    get_max_accel=get_max_accel, LeadLaunch=LeadLaunch,
    FinalStopTaper=FinalStopTaper, apply_min_gap_floor=apply_min_gap_floor,
    desired_bumper_gap=desired_bumper_gap, get_min_gap_floor=get_min_gap_floor,
    ChillAggressiveLaunch=ChillAggressiveLaunch, P=P,
    get_lead_stop_extra=get_lead_stop_extra, STOP_DISTANCE=STOP_DISTANCE,
    LongitudinalMpc=LongitudinalMpc, ACCEL_MAX=ACCEL_MAX,
    get_T_FOLLOW=get_T_FOLLOW, stopping_accel_step=stopping_accel_step,
  )
  out = {"note": "after = T=1.5 Agg, floor, gated launch/boost, chauffeur stop. "
                 "legacy = ungated launch (diagnosis). before_stop = official 1.0 hold, no taper.",
         "personalities": {}, "last5_before_after": {}, "cutin": {}, "brake": {}}

  # After-launch follow to 45 for all personalities (report 5-45)
  for name, p in (("aggressive", P.aggressive), ("standard", P.standard), ("relaxed", P.relaxed)):
    print(f"launch-follow after {name}...", file=sys.stderr)
    r = run_follow(p, "after", 45, kw, kind="launch")
    print(f"  min={r['min_gap_ft']} below={r['below_floor']} steady={r['steady_gap_ft']}/{r['steady_tg']}s T_eq={r['steady_T_eq']}", file=sys.stderr)
    print(f"  at_speed {r['at_speed']}", file=sys.stderr)
    # Steady at each speed 5-45
    steady = {}
    for mph in (5, 10, 15, 20, 25, 30, 35, 40, 45):
      s = run_follow(p, "after", mph, kw, kind="steady")
      steady[str(mph)] = {
        "gap_ft": s["steady_gap_ft"], "tg": s["steady_tg"],
        "T_eq": s["steady_T_eq"], "min_ft": s["min_gap_ft"],
        "below_floor": s["below_floor"], "T": s["T"] if mph == 5 else get_T_FOLLOW(p, False, mph*MPH),
      }
      print(f"  steady {mph}: {steady[str(mph)]}", file=sys.stderr)
    out["personalities"][name] = {"launch_to_45": r, "steady": steady}

  # Last-5-mph / hold: approach a stopping lead from 15 mph, before vs after, Agg
  # Reuse kind=brake from 15 mph
  for mode in ("before_stop", "after"):
    print(f"last5 {mode}...", file=sys.stderr)
    r = run_follow(P.aggressive, mode, 15, kw, kind="brake")
    print(f"  {r['last5']} min={r['min_gap_ft']} end_v={r['end_v_mph']} end_pa={r['end_pa']}", file=sys.stderr)
    out["last5_before_after"][mode] = {k: r[k] for k in
      ("min_gap_ft", "below_floor", "end_gap_ft", "end_v_mph", "end_pa", "last5", "steady_gap_ft")}
    # also Std and Rel after only
    if mode == "after":
      for name, p in (("standard", P.standard), ("relaxed", P.relaxed)):
        rr = run_follow(p, mode, 15, kw, kind="brake")
        out["last5_before_after"][f"after_{name}"] = {k: rr[k] for k in
          ("min_gap_ft", "below_floor", "end_gap_ft", "end_v_mph", "end_pa", "last5")}

  # Cut-in inside floor at 30 mph, all personalities, after
  for name, p in (("aggressive", P.aggressive), ("standard", P.standard), ("relaxed", P.relaxed)):
    print(f"cutin {name}...", file=sys.stderr)
    r = run_follow(p, "after", 30, kw, kind="cutin")
    print(f"  min={r['min_gap_ft']} below={r['below_floor']} end={r['end_gap_ft']}", file=sys.stderr)
    out["cutin"][name] = {k: r[k] for k in
      ("min_gap_ft", "below_floor", "t_below", "end_gap_ft", "end_v_mph", "floor_ft", "steady_gap_ft")}

  # Lead braking from 30, after, Agg
  print("brake agg...", file=sys.stderr)
  r = run_follow(P.aggressive, "after", 30, kw, kind="brake")
  print(f"  min={r['min_gap_ft']} below={r['below_floor']} end={r['end_gap_ft']} last5={r['last5']}", file=sys.stderr)
  out["brake"]["aggressive"] = {k: r[k] for k in
    ("min_gap_ft", "below_floor", "end_gap_ft", "end_v_mph", "end_pa", "last5", "floor_ft")}

  dest = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/workspace/handoff/devc4-test3-dev_floor_follow_sim.json")
  dest.write_text(json.dumps(out, indent=2, default=str))
  print(f"wrote {dest}")


if __name__ == "__main__":
  main()
