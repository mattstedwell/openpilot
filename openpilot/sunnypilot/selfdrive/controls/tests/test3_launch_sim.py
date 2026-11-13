#!/usr/bin/env python3
"""Host launch take-up sims for devc4-test3-dev.

usage: test3_launch_sim.py REPO [BUILD_DIR] [out.json]

Runs 1.0 and 0.5 m/s² pull-aways and a stopped-lead case, with and
without LeadLaunch, using this tree's C3X caps and +5/+5/+10 ft extras.
Compares t_req / t_move to the previous abab191b19 and C3X JSONs.
Nothing road-tested.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Import the equiv-sim helpers (they read REPO/BUILD from argv[1]/argv[2]
# at import; our invocation uses the same first two args).
from openpilot.sunnypilot.selfdrive.controls.tests.test3_c3x_equiv_sim import (
  _setup, _radar, _plan, BUILD, ROOT,
)


def sim_launch(personality, use_launch, a_lead, d0, get_max_accel,
               LeadLaunch, LAUNCH_JERK_UP, LAUNCH_START_STEP,
               ChillAggressiveLaunch, P, get_lead_stop_extra,
               STOP_DISTANCE, LongitudinalMpc,
               v_cruise=25.0, sim_t=12.0):
  import numpy as np

  dt_mdl, dt_ctrl, tau = 0.05, 0.01, 0.25
  MPH = 0.44704
  FT = 0.3048
  is_relaxed = personality == P.relaxed
  is_agg = personality == P.aggressive

  mpc = LongitudinalMpc()
  ls = LeadLaunch() if use_launch else None
  launch_boost_sm = ChillAggressiveLaunch()

  ex, ev, ea = 0.0, 0.0, 0.0
  lx, lv = d0, 0.0
  plan_a, v_des, last, state, ss = -0.5, 0.0, -0.5, "stopping", True
  a_cruise_st = plan_a
  t_req_pos = t_req_005 = t_move_005 = t_move_1mph = None
  peak_a, peak_j, min_gap = -1e9, 0.0, 1e9
  peak_a_1s, peak_j_1s, peak_j_1s_ex = -1e9, 0.0, 0.0
  peak_plant_j_1s = 0.0
  prev_a = plan_a
  prev_ea = 0.0
  artifact_frames = 0
  final_gap = final_v = None
  crept = False

  for k in range(int(sim_t / dt_ctrl)):
    t = k * dt_ctrl
    la = a_lead if t >= 0.0 else 0.0
    lv = max(lv + la * dt_ctrl, 0.0)
    lx += lv * dt_ctrl
    if k % 5 == 0:
      d_rel = lx - ex
      present = True
      rs = _radar(d_rel, lv, la, present=present)
      rs.leadOne.vRel = float(lv - ev)
      # Chill Aggressive +0.4 cap boost: arms only at a stop, same as the planner.
      launch_active = launch_boost_sm.update(
        True, False, personality, ev, v_cruise, False)
      cap = float(get_max_accel(ev, launch_active, is_relaxed))
      a_t, ss, a_cruise_st = _plan(
        mpc, ev, plan_a, v_des, rs, v_cruise, personality, False, cap,
        a_cruise_prev=a_cruise_st)
      if a_t > 0:
        a_t = min(a_t, cap)
      if use_launch:
        a_t, ss = ls.apply(
          a_t, ss, ev, v_cruise, rs.leadOne, False, dt_mdl,
          cap=cap, jerk_up=LAUNCH_JERK_UP, v_ego_stopping=0.25)
        a_t = min(a_t, cap) if a_t > 0 else a_t
      jerk = abs(a_t - prev_a) / dt_mdl
      peak_j = max(peak_j, jerk)
      if t <= 1.0:
        peak_j_1s = max(peak_j_1s, jerk)
        # Preview (0.10) and 0.30 first step are one-frame command
        # artifacts (Δa / 0.05). Felt take-up is the 0.4 m/s³ ramp.
        is_artifact = False
        if a_t > 0.0 and prev_a < LAUNCH_START_STEP - 1e-9 and artifact_frames < 2:
          is_artifact = True
          artifact_frames += 1
        if not is_artifact:
          peak_j_1s_ex = max(peak_j_1s_ex, jerk)
      prev_a = a_t
      plan_a = float(np.clip(a_t, -3.5, 2.5))
      v_des = 0.5 * max(0.0, v_des + dt_mdl * plan_a) + 0.5 * ev
      if ev < 0.25:
        ss = plan_a < 0.1
    # official sunnypilot/dev longcontrol stopping ramp is 1.0 (same plant
    # as the previous test3-dev equiv sims, so before/after is fair).
    if state == "stopping" and (not ss):
      state = "pid"
    elif state == "pid" and ss and ev < 0.25:
      state = "stopping"
    if state == "stopping":
      out = last
      if out > -2.0:
        out = min(out, 0.0) - 1.0 * dt_ctrl
      out = max(out, -2.0)
    else:
      out = float(ea + (plan_a - ea) * (1.0 - np.exp(-dt_ctrl / tau)))
    last = out
    ea = out
    ev = max(0.0, ev + ea * dt_ctrl)
    if ev <= 0.0 and ea < 0.0:
      ev = 0.0
    ex += ev * dt_ctrl
    gap = lx - ex
    min_gap = min(min_gap, gap)
    final_gap = gap
    final_v = ev
    peak_a = max(peak_a, plan_a)
    if t <= 1.0:
      peak_a_1s = max(peak_a_1s, plan_a)
      peak_plant_j_1s = max(peak_plant_j_1s, abs(ea - prev_ea) / dt_ctrl)
    prev_ea = ea
    if t_req_pos is None and plan_a > 0.0:
      t_req_pos = t
    if t_req_005 is None and plan_a > 0.05:
      t_req_005 = t
    if t_move_005 is None and ev > 0.05:
      t_move_005 = t
    if t_move_1mph is None and ev > MPH:
      t_move_1mph = t
    if ev > 0.05 and a_lead == 0.0:
      crept = True

  return {
    "t_req_pos": None if t_req_pos is None else round(t_req_pos, 3),
    "t_req_005": None if t_req_005 is None else round(t_req_005, 3),
    "t_move_005": None if t_move_005 is None else round(t_move_005, 3),
    "t_move_1mph": None if t_move_1mph is None else round(t_move_1mph, 3),
    # aliases matching the test2-dev report (ev > 0.05, first plan_a > 0)
    "t_accel_req_s": None if t_req_pos is None else round(t_req_pos, 3),
    "t_move_s": None if t_move_005 is None else round(t_move_005, 3),
    "peak_accel": round(float(peak_a), 4),
    "peak_jerk": round(float(peak_j), 4),
    "peak_accel_1s": round(float(peak_a_1s), 4) if peak_a_1s > -1e8 else None,
    "peak_jerk_1s": round(float(peak_j_1s), 4),
    "peak_jerk_1s_ex_artifact": round(float(peak_j_1s_ex), 4),
    "plant_peak_jerk_1s": round(float(peak_plant_j_1s), 4),
    "min_gap_m": round(float(min_gap), 4),
    "min_gap_ft": round(float(min_gap) / FT, 2),
    "final_gap_m": None if final_gap is None else round(float(final_gap), 4),
    "end_v": None if final_v is None else round(float(final_v), 4),
    "d0_m": round(d0, 4),
    "d0_ft": round(d0 / FT, 2),
    "crept": bool(crept) if a_lead == 0.0 else False,
    "cap_at_stop": float(get_max_accel(0.0, is_agg, is_relaxed)),
  }


def main():
  from openpilot.cereal import log
  g = _setup(BUILD, ROOT)
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
    LongitudinalMpc, get_lead_stop_extra, STOP_DISTANCE,
  )
  from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel
  from openpilot.sunnypilot.selfdrive.car.chill_aggressive import ChillAggressiveLaunch
  from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_launch import (
    LeadLaunch, LAUNCH_JERK_UP, LAUNCH_START_STEP,
  )

  P = log.LongitudinalPersonality
  personalities = {
    "aggressive": P.aggressive,
    "standard": P.standard,
    "relaxed": P.relaxed,
  }
  FT = 0.3048

  def d0_for(p):
    return STOP_DISTANCE + get_lead_stop_extra(p, False)

  results = {"after": {}, "before": {}}
  for use, label in ((True, "after"), (False, "before")):
    for name, p in personalities.items():
      d0 = d0_for(p)
      for a_lead, tag in ((1.0, "pullaway_1p0"), (0.5, "pullaway_0p5"), (0.0, "stopped")):
        key = f"launch_{tag}_{name}"
        results[label][key] = sim_launch(
          p, use, a_lead, d0, get_max_accel,
          LeadLaunch, LAUNCH_JERK_UP, LAUNCH_START_STEP,
          ChillAggressiveLaunch, P, get_lead_stop_extra,
          STOP_DISTANCE, LongitudinalMpc,
        )
        results[label][key]["a_lead"] = a_lead
        results[label][key]["personality"] = name

  # Baselines from previous handoff JSONs (1.0 pull-away only; 1 mph t_move).
  baselines = {}
  for path, lab in (
    ("/workspace/handoff/devc4-test3-dev_c3xcaps_equiv.json", "abab191b19_c3xcaps"),
    ("/workspace/handoff/devc4-test3-dev_equiv.json", "6b8de3eced_official_2p0"),
    ("/workspace/handoff/devc4-test3_equiv_c3x.json", "c3x"),
    ("/workspace/handoff/devc4-test3_equiv_test3.json", "old_test3"),
  ):
    pth = Path(path)
    if not pth.exists():
      continue
    data = json.loads(pth.read_text())
    # c3xcaps file is a comparison wrapper with 'sims' at top or nested
    sims = data.get("sims") or data.get("policy") and data  # raw style
    if "sims" in data:
      src = data["sims"]
    elif "policy" in data and "sims" not in data:
      src = {}
    else:
      src = data.get("sims", {})
    entry = {}
    for name in personalities:
      k = f"launch_{name}"
      if k in src:
        s = src[k]
        entry[name] = {
          "t_req": s.get("t_req"),
          "t_move": s.get("t_move"),
          "peak_a": s.get("peak_a"),
          "peak_j": s.get("peak_j"),
          "min_gap": s.get("min_gap"),
          "note": "previous equiv sim: t_req is plan_a>0.05, t_move is ev>1 mph",
        }
    baselines[lab] = entry

  # Compact comparison table
  comparison = {}
  for name in personalities:
    comparison[name] = {}
    for a_tag, a_lead in (("pullaway_1p0", 1.0), ("pullaway_0p5", 0.5)):
      after = results["after"][f"launch_{a_tag}_{name}"]
      before = results["before"][f"launch_{a_tag}_{name}"]
      comparison[name][a_tag] = {
        "after": {
          "t_req_pos": after["t_req_pos"],
          "t_move_005": after["t_move_005"],
          "t_move_1mph": after["t_move_1mph"],
          "peak_accel": after["peak_accel"],
          "peak_jerk": after["peak_jerk"],
          "peak_jerk_1s_ex_artifact": after["peak_jerk_1s_ex_artifact"],
          "min_gap_m": after["min_gap_m"],
          "min_gap_ft": after["min_gap_ft"],
        },
        "before_same_plant": {
          "t_req_pos": before["t_req_pos"],
          "t_req_005": before["t_req_005"],
          "t_move_005": before["t_move_005"],
          "t_move_1mph": before["t_move_1mph"],
          "peak_accel": before["peak_accel"],
          "peak_jerk": before["peak_jerk"],
          "min_gap_m": before["min_gap_m"],
          "min_gap_ft": before["min_gap_ft"],
        },
      }
    # stopped
    sa = results["after"][f"launch_stopped_{name}"]
    sb = results["before"][f"launch_stopped_{name}"]
    comparison[name]["stopped"] = {
      "after_crept": sa["crept"],
      "after_t_move_005": sa["t_move_005"],
      "after_peak_accel": sa["peak_accel"],
      "after_min_gap_m": sa["min_gap_m"],
      "after_final_gap_m": sa["final_gap_m"],
      "before_crept": sb["crept"],
      "before_t_move_005": sb["t_move_005"],
      "planned_gap_m": sa["d0_m"],
    }

  out = {
    "repo": str(ROOT),
    "note": (
      "Launch take-up only (test2 LaunchStart + predictive go / 0.10 preview). "
      "C3X caps; test3 +5/+5/+10 ft extras; no LeadDecel, no +8 ft, no 30%-PWR. "
      "t_req_pos = first plan_a>0 (test2 style); t_req_005 = plan_a>0.05 (old test3 equiv); "
      "t_move_005 = ev>0.05 (test2 0.35 s number); t_move_1mph = ev>0.447 (old test3/C3X report). "
      "peak_jerk raw includes one-frame preview/step artifacts; "
      "peak_jerk_1s_ex_artifact is the felt 0.4 m/s³ ramp. "
      "Official stopping ramp 1.0, same plant as previous test3-dev equiv. Not road-tested."
    ),
    "caps_at_stop": {
      "aggressive": float(get_max_accel(0.0, True, False)),
      "standard": float(get_max_accel(0.0, False, False)),
      "relaxed": float(get_max_accel(0.0, False, True)),
    },
    "planned_standstill_ft": {
      name: round((STOP_DISTANCE + get_lead_stop_extra(p, False)) / FT, 2)
      for name, p in personalities.items()
    },
    "comparison": comparison,
    "baselines_previous_equiv_1p0_1mph": baselines,
    "after": results["after"],
    "before": results["before"],
  }
  print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
  import io
  from contextlib import redirect_stdout
  args = [a for a in sys.argv[1:] if a.endswith(".json")]
  dest = args[-1] if args else os.environ.get("TEST3_LAUNCH_OUT")
  buf = io.StringIO()
  with redirect_stdout(buf):
    main()
  text = buf.getvalue()
  i = text.find("{")
  if i < 0:
    sys.stderr.write(text)
    sys.exit(2)
  text = text[i:]
  if dest:
    Path(dest).write_text(text)
    sys.stderr.write(f"wrote {dest} ({len(text)} bytes)\n")
  else:
    sys.stdout.write(text)
