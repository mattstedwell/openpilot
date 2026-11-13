#!/usr/bin/env python3
"""devc4-test3 launch-bypass-fix closed-loop sims.

usage: test3_launchfix_sim.py [REPO] [BUILD_DIR] [out.json]

Stop behind a lead; lead accelerates at 1.0 m/s² to 45 mph and holds.
Reports gaps (ft and s) at 15/25/30/45 mph, t_req / t_move, min gap,
whether launch releases, and whether the floor was violated.
Import-safe (no argv at module import) so the pytest regression can
reuse run(). Nothing road-tested.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MPH = 0.44704
FT = 0.3048


def worktree_root() -> Path:
  here = Path(__file__).resolve()
  # .../openpilot/sunnypilot/selfdrive/controls/tests/this.py
  return here.parents[5]


def run_launch_follow(personality, kw, target_mph=45, sim_t=45.0, use_launch=True):
  """Stop behind lead; lead accel 1.0 to target_mph and holds."""
  import numpy as np
  get_max_accel = kw["get_max_accel"]
  LeadLaunch = kw["LeadLaunch"]
  desired_follow_gap = kw["desired_follow_gap"]
  ChillAggressiveLaunch = kw["ChillAggressiveLaunch"]
  P = kw["P"]
  get_lead_stop_extra = kw["get_lead_stop_extra"]
  STOP_DISTANCE = kw["STOP_DISTANCE"]
  LongitudinalMpc = kw["LongitudinalMpc"]
  get_T_FOLLOW = kw["get_T_FOLLOW"]
  _radar = kw["_radar"]
  _plan = kw["_plan"]

  dt_mdl, dt_ctrl, tau = 0.05, 0.01, 0.25
  is_relaxed = personality == P.relaxed
  is_agg = personality == P.aggressive
  floor = STOP_DISTANCE + get_lead_stop_extra(personality, False)
  v_tgt = target_mph * MPH
  v_cruise = 25.0

  mpc = LongitudinalMpc()
  ls = LeadLaunch() if use_launch else None
  boost_sm = ChillAggressiveLaunch()

  ex, ev, ea = 0.0, 0.0, 0.0
  lx, lv = floor, 0.0
  plan_a, v_des, last, state, ss = -0.5, 0.0, -0.5, "stopping", True
  a_cruise_st = plan_a
  at_speed = {}
  min_gap = 1e9
  t_req = t_move = t_move_1mph = None
  launch_last_on = None
  launch_first_off = None
  launch_on = False
  traces = []

  for k in range(int(sim_t / dt_ctrl)):
    t = k * dt_ctrl
    la = 0.0 if lv >= v_tgt else 1.0
    if lv < v_tgt and lv + la * dt_ctrl > v_tgt:
      la = (v_tgt - lv) / dt_ctrl
    lv = min(max(lv + la * dt_ctrl, 0.0), v_tgt)
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
      if a_t > 0:
        a_t = min(a_t, cap)
      if use_launch:
        desired = desired_follow_gap(ev, lv, personality, False)
        a_t, ss = ls.apply(
          a_t, ss, ev, v_cruise, rs.leadOne, False, dt_mdl,
          cap=cap, v_ego_stopping=0.25, desired_gap=desired)
        if a_t > 0:
          a_t = min(a_t, cap)
        launch_on = bool(ls.launch.active or ls.launch.go_count > 0)
        if launch_on:
          launch_last_on = t
        elif launch_last_on is not None and launch_first_off is None:
          launch_first_off = t
      plan_a = float(np.clip(a_t, -3.5, 2.5))
      v_des = 0.5 * max(0.0, v_des + dt_mdl * plan_a) + 0.5 * ev
      if ev < 0.25:
        ss = plan_a < 0.1
      if t_req is None and plan_a > 0.0:
        t_req = t

    if state == "stopping" and (not ss):
      state = "pid"
    elif state == "pid" and ss and ev < 0.25:
      state = "stopping"
    if state == "stopping":
      out = last
      if out > -2.0:
        out = min(out, 0.0) - 1.0 * dt_ctrl  # official 1.0 ramp (unchanged)
      out = max(out, -2.0)
    else:
      out = float(ea + (plan_a - ea) * (1.0 - np.exp(-dt_ctrl / tau)))
    last = out
    ea = out
    ev = max(0.0, ev + ea * dt_ctrl)
    if ev <= 0 and ea < 0:
      ev = 0.0
    ex += ev * dt_ctrl
    gap = lx - ex
    min_gap = min(min_gap, gap)
    if t_move is None and ev > 0.05:
      t_move = t
    if t_move_1mph is None and ev > MPH:
      t_move_1mph = t
    ev_mph = ev / MPH
    T = get_T_FOLLOW(personality, False, ev)
    desired = T * ev + STOP_DISTANCE  # extras out at speed
    tg = gap / ev if ev > 0.3 else None
    t_eq = (gap - STOP_DISTANCE) / ev if ev > 0.3 else None
    for mph in (5, 10, 15, 20, 25, 30, 35, 40, 45):
      if mph not in at_speed and ev >= mph * MPH:
        at_speed[mph] = {
          "t": round(t, 2),
          "gap_ft": round(gap / FT, 1),
          "time_gap_s": None if tg is None else round(tg, 2),
          "T_eq": None if t_eq is None else round(t_eq, 2),
          "desired_ft": round(desired / FT, 1),
          "launch_active": launch_on,
          "below_floor": gap < floor - 0.02,
        }
    if k % 50 == 0:
      traces.append({
        "t": round(t, 2), "ev_mph": round(ev_mph, 2),
        "gap_ft": round(gap / FT, 1),
        "tg": None if tg is None else round(tg, 2),
        "T_eq": None if t_eq is None else round(t_eq, 2),
        "des": round(desired / FT, 1),
        "lnch": launch_on, "pa": round(plan_a, 3),
      })

  tail = [x for x in traces if x["t"] >= sim_t - 3]

  def avg(key):
    xs = [x[key] for x in tail if x.get(key) is not None]
    return None if not xs else round(sum(xs) / len(xs), 2)

  return {
    "T_at_30": get_T_FOLLOW(personality, False, 30 * MPH),
    "floor_ft": round(floor / FT, 2),
    "t_req": None if t_req is None else round(t_req, 2),
    "t_move": None if t_move is None else round(t_move, 2),
    "t_move_1mph": None if t_move_1mph is None else round(t_move_1mph, 2),
    "at_speed": {str(k): at_speed[k] for k in sorted(at_speed)},
    "min_gap_ft": round(min_gap / FT, 1),
    "below_floor": min_gap < floor - 0.02,
    "steady_gap_ft": avg("gap_ft"),
    "steady_tg": avg("tg"),
    "steady_T_eq": avg("T_eq"),
    "end_gap_ft": traces[-1]["gap_ft"] if traces else None,
    "launch_last_on_s": None if launch_last_on is None else round(launch_last_on, 2),
    "launch_first_off_s": None if launch_first_off is None else round(launch_first_off, 2),
    "launch_on_at_end": traces[-1]["lnch"] if traces else None,
  }


def prepare(repo=None, build=None):
  """Install the host MPC solver and return kw for run_launch_follow.

  Import-safe: pass explicit paths. Temporarily pins argv so the
  equiv-sim module (which reads argv at import) finds the worktree.
  """
  repo = Path(repo) if repo else worktree_root()
  build = Path(build) if build else Path("/tmp/test3dev_mpc_build")
  old = sys.argv[:]
  sys.argv = [old[0], str(repo), str(build)]
  try:
    # Fresh import if a previous pytest import used a bad argv.
    for name in list(sys.modules):
      if name.endswith("test3_c3x_equiv_sim"):
        del sys.modules[name]
    from openpilot.sunnypilot.selfdrive.controls.tests.test3_c3x_equiv_sim import (
      _setup, _radar, _plan, BUILD, ROOT,
    )
  finally:
    sys.argv = old
  _setup(build, repo)
  # After _setup installs the real host AcadosOcpSolverCython, reload
  # long_mpc so a prior pytest host-stub bind cannot stick.
  import importlib
  name = "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc"
  if name in sys.modules:
    importlib.reload(sys.modules[name])
  from openpilot.cereal import log
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
    LongitudinalMpc, get_lead_stop_extra, get_T_FOLLOW, STOP_DISTANCE,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel
  from openpilot.sunnypilot.selfdrive.car.chill_aggressive import ChillAggressiveLaunch
  from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_launch import (
    LeadLaunch, desired_follow_gap,
  )
  return dict(
    get_max_accel=get_max_accel, LeadLaunch=LeadLaunch,
    desired_follow_gap=desired_follow_gap,
    ChillAggressiveLaunch=ChillAggressiveLaunch, P=log.LongitudinalPersonality,
    get_lead_stop_extra=get_lead_stop_extra, STOP_DISTANCE=STOP_DISTANCE,
    LongitudinalMpc=LongitudinalMpc, get_T_FOLLOW=get_T_FOLLOW,
    _radar=_radar, _plan=_plan,
  )


def main(repo=None, build=None, out=None):
  kw = prepare(repo, build)
  P = kw["P"]
  result = {"note": "launch-bypass fix only: already_active capped at ~11 mph, "
                    "take-up gated on gap >= desired. T_FOLLOW Agg=1.25, official "
                    "1.0 stopping ramp, no min-gap-floor hook, no chauffeur taper.",
            "personalities": {}}
  for name, p in (("aggressive", P.aggressive), ("standard", P.standard),
                  ("relaxed", P.relaxed)):
    print(f"launch-follow {name}...", file=sys.stderr)
    r = run_launch_follow(p, kw, target_mph=45, sim_t=45.0)
    print(f"  t_req={r['t_req']} t_move={r['t_move']} t_move_1mph={r['t_move_1mph']} "
          f"min={r['min_gap_ft']} below={r['below_floor']} "
          f"steady={r['steady_gap_ft']}/{r['steady_tg']}s T_eq={r['steady_T_eq']} "
          f"launch_off={r['launch_first_off_s']} on_end={r['launch_on_at_end']}",
          file=sys.stderr)
    for mph in ("15", "25", "30", "45"):
      print(f"    {mph}: {r['at_speed'].get(mph)}", file=sys.stderr)
    result["personalities"][name] = r
  dest = Path(out) if out else Path("/workspace/handoff/devc4-test3-dev_launchfix_sim.json")
  dest.write_text(json.dumps(result, indent=2, default=float))
  print(f"wrote {dest}")
  return result


if __name__ == "__main__":
  repo = sys.argv[1] if len(sys.argv) > 1 else None
  build = sys.argv[2] if len(sys.argv) > 2 else None
  out = sys.argv[3] if len(sys.argv) > 3 else None
  main(repo, build, out)
