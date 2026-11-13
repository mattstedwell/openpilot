#!/usr/bin/env python3
"""Closed-loop host sims for devc4-test3 chauffeur final stop and hold.

usage: test3_final_stop_sim.py [REPO] [BUILD_DIR] [out.json]

Stops from 25 mph and from 5 mph, with a stopping lead and with a
lead already at rest, plus a no-lead cruise-to-zero and an emergency
(tight TTC) case. Reports peak |jerk| in the last 1 mph and during
the hold transition, versus the current-tip "before" path (official
1.0 m/s²/s longcontrol ramp, no taper). Stopping-distance change and
emergency decel are also reported.

Import-safe (no argv at module import). Nothing road-tested.
Plant tau is 0.25. Report planner and plant (felt) jerk, not the
0.01 s command-sample artifact at a state switch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MPH = 0.44704
FT = 0.3048
LAST_1MPH = 1.0 * MPH  # 0.447 m/s
STOP_ACCEL = -2.0


def worktree_root() -> Path:
  here = Path(__file__).resolve()
  return here.parents[5]


def _plant_jerk_sample(j, ev, t, ignore_first=0.15):
  """Drop the first settle and one-frame state-switch spikes (> 4)."""
  if t <= ignore_first:
    return False
  if j > 4.0:
    return False
  return True


def sim_stop(personality, after, kind, v0_mph, kw, a_lead=-1.5, sim_t=30.0,
             emergency=False):
  """kind:
    'stopping_lead' — both at v0, lead brakes at a_lead from t=0.5
    'stopped_lead'  — lead already at rest, ego approaches from v0
    'no_lead'       — no lead, cruise set to 0 (always-Chill cruise stop)
    'emergency'     — tight gap, lead already stopped, full braking
  """
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
  from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_final_stop import (
    FinalStopTaper, stopping_accel_step, FINAL_STOP_JERK, FINAL_STOP_REST_DECEL,
  )

  dt_mdl, dt_ctrl, tau = 0.05, 0.01, 0.25
  is_relaxed = personality == P.relaxed
  is_agg = personality == P.aggressive
  v0 = v0_mph * MPH
  T = get_T_FOLLOW(personality, False, v0)
  extra = get_lead_stop_extra(personality, False)
  target = STOP_DISTANCE + extra  # planned standstill (extras on at rest)
  present = True
  t_brake = 0.5
  if kind == "stopping_lead":
    d0 = T * v0 + target
    lv0, ev0 = v0, v0
  elif kind == "stopped_lead":
    # Room for a comfortable (~1.0 m/s²) stop plus the standstill
    # target and a margin, so we are measuring the last-mph finish,
    # not a last-second panic.
    d0 = (v0 * v0) / (2.0 * 1.0) + target + 8.0
    lv0, ev0 = 0.0, v0
    a_lead = 0.0
    t_brake = 0.0
  elif kind == "no_lead":
    d0 = 200.0
    lv0, ev0 = 0.0, v0
    a_lead = 0.0
    t_brake = 0.0
    present = False
  elif kind == "emergency":
    # Tight gap, lead already stopped: TTC at 25 mph / 8 m ≈ 0.7 s.
    d0 = 8.0 if v0_mph >= 15 else 4.0
    lv0, ev0 = 0.0, v0
    a_lead = 0.0
    t_brake = 0.0
    emergency = True
  else:
    raise ValueError(kind)

  mpc = LongitudinalMpc()
  mpc.final_stop_mpc = bool(after)
  ls = LeadLaunch()
  boost_sm = ChillAggressiveLaunch()
  taper = FinalStopTaper() if after else None
  v_cruise = 0.0 if kind == "no_lead" else max(v0 + 5.0, 25.0)

  ex, ev, ea = 0.0, ev0, 0.0
  lx, lv = d0, lv0
  plan_a, v_des, last, state, ss = 0.0, ev0, 0.0, "pid", False
  a_cruise_st = plan_a
  prev_plan = 0.0
  prev_out = 0.0
  prev_ea = 0.0
  peak_decel = 0.0
  peak_j_plan = 0.0
  peak_j_out = 0.0
  peak_j_plant = 0.0
  peak_j_plan_last1 = 0.0
  peak_j_out_last1 = 0.0
  peak_j_plant_last1 = 0.0
  peak_decel_last1 = 0.0
  standstill_step = None
  standstill_jerk = None
  standstill_out_before = None
  standstill_out_after = None
  hold_peak_j_out = 0.0
  hold_peak_j_plant = 0.0
  t_standstill = None
  saw_standstill = False
  min_gap = 1e9
  t_stop = None
  t_first_brake = None
  t_5mph = None
  t_1mph = None
  dist_from_5 = None
  x_at_5 = None
  a_at_8mph = None
  a_at_5mph = None
  a_at_1mph = None
  a_at_0 = None
  final_gap = None
  last1_outs = []
  creep = False

  for k in range(int(sim_t / dt_ctrl)):
    t = k * dt_ctrl
    if kind == "stopping_lead":
      la = a_lead if (t >= t_brake and lv > 0.0) else 0.0
    else:
      la = 0.0
    lv = max(lv + la * dt_ctrl, 0.0)
    lx += lv * dt_ctrl

    if k % 5 == 0:
      d_rel = lx - ex
      rs = _radar(d_rel, lv, la, present=present)
      if present:
        rs.leadOne.vRel = float(lv - ev)
      launch_boost = boost_sm.update(True, False, personality, ev, v_cruise, False)
      cap = float(get_max_accel(ev, launch_boost if is_agg else False, is_relaxed))
      a_t, ss, a_cruise_st = _plan(
        mpc, ev, plan_a, v_des, rs, v_cruise, personality, False, cap,
        a_cruise_prev=a_cruise_st)
      if a_t > 0:
        a_t = min(a_t, cap)
      desired = desired_follow_gap(ev, lv, personality, False) if present else None
      a_t, ss = ls.apply(
        a_t, ss, ev, v_cruise, rs.leadOne, False, dt_mdl,
        cap=cap, v_ego_stopping=0.25, desired_gap=desired)
      if a_t > 0:
        a_t = min(a_t, cap)
      if after:
        a_t = taper.apply(
          a_t, ev, rs.leadOne if present else None, False, dt_mdl,
          should_stop=bool(ss), a_prev=plan_a)
      plan_a = float(np.clip(a_t, -3.5, 2.5))
      v_des = 0.5 * max(0.0, v_des + dt_mdl * plan_a) + 0.5 * ev
      if ev < 0.3:
        ss = plan_a < 0.1
      j_plan = abs(plan_a - prev_plan) / dt_mdl
      if _plant_jerk_sample(j_plan, ev, t):
        peak_j_plan = max(peak_j_plan, j_plan)
        if 0.02 < ev <= LAST_1MPH:
          peak_j_plan_last1 = max(peak_j_plan_last1, j_plan)
      prev_plan = plan_a

    # longcontrol state: official should_stop is v < 0.3 and a < 0.1
    if state == "pid" and ss and ev < 0.3:
      state = "stopping"
    elif state == "stopping" and (not ss) and ev > 0.3:
      state = "pid"

    if state == "stopping":
      out = last
      if out > STOP_ACCEL:
        if after:
          standstill = ev <= 0.02
          out = stopping_accel_step(out, standstill, dt_ctrl, ev, stop_accel=STOP_ACCEL)
          out = max(min(out, 0.0), STOP_ACCEL)
        else:
          # Official sunnypilot/dev: build toward stopAccel at 1.0
          # m/s²/s even while rolling — the last-mph lurch.
          out = min(out, 0.0) - 1.0 * dt_ctrl
          out = max(out, STOP_ACCEL)
    else:
      out = float(ea + (plan_a - ea) * (1.0 - np.exp(-dt_ctrl / tau)))

    # Standstill transition: first frame ev hits ~0
    if (not saw_standstill) and ev <= 0.02 and prev_ea is not None and t > 0.1:
      if ev + (-min(out, 0.0)) * dt_ctrl > 0.0 or ev == 0.0 or (ev < 0.05 and out < 0):
        # capture on first crossing to ev<=0.02 while commanding brake
        if ev <= 0.02 and out <= 0.0:
          saw_standstill = True
          t_standstill = t
          standstill_out_before = prev_out
          standstill_out_after = out
          standstill_step = out - prev_out
          standstill_jerk = abs(standstill_step) / dt_ctrl

    j_out = abs(out - prev_out) / dt_ctrl
    if _plant_jerk_sample(j_out, ev, t):
      peak_j_out = max(peak_j_out, j_out)
      if 0.02 < ev <= LAST_1MPH:
        peak_j_out_last1 = max(peak_j_out_last1, j_out)
      if saw_standstill and t_standstill is not None and t_standstill < t <= t_standstill + 2.0:
        hold_peak_j_out = max(hold_peak_j_out, j_out)

    last = out
    ea = out
    ev_prev = ev
    ev = max(0.0, ev + ea * dt_ctrl)
    if ev <= 0 and ea < 0:
      ev = 0.0
    ex += ev * dt_ctrl
    if ev > 0.02 and ea > 0.02 and t_stop is not None:
      creep = True

    j_plant = abs(ea - prev_ea) / dt_ctrl
    if _plant_jerk_sample(j_plant, ev, t):
      peak_j_plant = max(peak_j_plant, j_plant)
      if 0.02 < ev <= LAST_1MPH:
        peak_j_plant_last1 = max(peak_j_plant_last1, j_plant)
      if saw_standstill and t_standstill is not None and t_standstill < t <= t_standstill + 2.0:
        hold_peak_j_plant = max(hold_peak_j_plant, j_plant)
    prev_ea = ea
    prev_out = out

    peak_decel = min(peak_decel, ea)
    if 0.02 < ev <= LAST_1MPH:
      peak_decel_last1 = min(peak_decel_last1, ea)
      last1_outs.append(round(float(out), 3))

    if present:
      gap = lx - ex
      min_gap = min(min_gap, gap)
      final_gap = gap
    else:
      final_gap = None
      min_gap = None

    if t_first_brake is None and plan_a < -0.05:
      t_first_brake = t
    if t_5mph is None and ev_prev > 5 * MPH >= ev:
      t_5mph = t
      x_at_5 = ex
      a_at_5mph = ea
    if t_1mph is None and ev_prev > LAST_1MPH >= ev:
      t_1mph = t
      a_at_1mph = ea
    if a_at_8mph is None and ev_prev > 8 * MPH >= ev:
      a_at_8mph = ea
    if t_stop is None and ev <= 0.02 and ev_prev > 0.02:
      t_stop = t
      a_at_0 = ea
      if x_at_5 is not None:
        dist_from_5 = ex - x_at_5
    if ev <= 0.02 and t_stop is not None and t > t_stop + 1.0:
      # settled; can end early once hold has been observed
      if t > t_stop + 3.0:
        break

  # distance traveled from v0 to rest
  dist_total = ex

  return {
    "kind": kind,
    "v0_mph": v0_mph,
    "after": after,
    "personality": int(personality),
    "emergency": emergency,
    "T": round(float(T), 3),
    "planned_standstill_ft": round(target / FT, 2),
    "d0_ft": round(d0 / FT, 1),
    "t_first_brake": None if t_first_brake is None else round(t_first_brake, 2),
    "t_5mph": None if t_5mph is None else round(t_5mph, 2),
    "t_1mph": None if t_1mph is None else round(t_1mph, 2),
    "t_stop": None if t_stop is None else round(t_stop, 2),
    "t_standstill": None if t_standstill is None else round(t_standstill, 2),
    "dist_total_m": round(float(dist_total), 3),
    "dist_from_5mph_m": None if dist_from_5 is None else round(float(dist_from_5), 3),
    "dist_from_5mph_ft": None if dist_from_5 is None else round(float(dist_from_5) / FT, 2),
    "final_gap_ft": None if final_gap is None else round(float(final_gap) / FT, 2),
    "min_gap_ft": None if min_gap is None else round(float(min_gap) / FT, 2),
    "peak_decel": round(float(peak_decel), 3),
    "peak_decel_last1mph": round(float(peak_decel_last1), 3),
    "peak_jerk_plan": round(float(peak_j_plan), 3),
    "peak_jerk_cmd": round(float(peak_j_out), 3),
    "peak_jerk_plant": round(float(peak_j_plant), 3),
    "peak_jerk_plan_last1mph": round(float(peak_j_plan_last1), 3),
    "peak_jerk_cmd_last1mph": round(float(peak_j_out_last1), 3),
    "peak_jerk_plant_last1mph": round(float(peak_j_plant_last1), 3),
    "standstill_step": None if standstill_step is None else round(float(standstill_step), 3),
    "standstill_jerk": None if standstill_jerk is None else round(float(standstill_jerk), 3),
    "standstill_out_before": None if standstill_out_before is None else round(float(standstill_out_before), 3),
    "standstill_out_after": None if standstill_out_after is None else round(float(standstill_out_after), 3),
    "hold_peak_jerk_cmd": round(float(hold_peak_j_out), 3),
    "hold_peak_jerk_plant": round(float(hold_peak_j_plant), 3),
    "a_at_8mph": None if a_at_8mph is None else round(float(a_at_8mph), 3),
    "a_at_5mph": None if a_at_5mph is None else round(float(a_at_5mph), 3),
    "a_at_1mph": None if a_at_1mph is None else round(float(a_at_1mph), 3),
    "a_at_0": None if a_at_0 is None else round(float(a_at_0), 3),
    "rest_decel_design": FINAL_STOP_REST_DECEL,
    "design_jerk": FINAL_STOP_JERK,
    "creep": creep,
    "stopped": t_stop is not None,
  }


def prepare(repo=None, build=None):
  from openpilot.sunnypilot.selfdrive.controls.tests.test3_launchfix_sim import (
    prepare as _prep,
  )
  return _prep(repo, build)


def compare(before, after):
  def d(key):
    b, a = before.get(key), after.get(key)
    if b is None or a is None:
      return None
    return round(float(a) - float(b), 3)

  return {
    "jerk_plant_last1mph_before": before.get("peak_jerk_plant_last1mph"),
    "jerk_plant_last1mph_after": after.get("peak_jerk_plant_last1mph"),
    "jerk_plant_last1mph_delta": d("peak_jerk_plant_last1mph"),
    "jerk_cmd_last1mph_before": before.get("peak_jerk_cmd_last1mph"),
    "jerk_cmd_last1mph_after": after.get("peak_jerk_cmd_last1mph"),
    "jerk_cmd_last1mph_delta": d("peak_jerk_cmd_last1mph"),
    "hold_jerk_cmd_before": before.get("hold_peak_jerk_cmd"),
    "hold_jerk_cmd_after": after.get("hold_peak_jerk_cmd"),
    "hold_jerk_cmd_delta": d("hold_peak_jerk_cmd"),
    "hold_jerk_plant_before": before.get("hold_peak_jerk_plant"),
    "hold_jerk_plant_after": after.get("hold_peak_jerk_plant"),
    "standstill_jerk_before": before.get("standstill_jerk"),
    "standstill_jerk_after": after.get("standstill_jerk"),
    "a_at_1mph_before": before.get("a_at_1mph"),
    "a_at_1mph_after": after.get("a_at_1mph"),
    "a_at_0_before": before.get("a_at_0"),
    "a_at_0_after": after.get("a_at_0"),
    "final_gap_ft_before": before.get("final_gap_ft"),
    "final_gap_ft_after": after.get("final_gap_ft"),
    "final_gap_ft_delta": d("final_gap_ft"),
    "dist_from_5mph_ft_before": before.get("dist_from_5mph_ft"),
    "dist_from_5mph_ft_after": after.get("dist_from_5mph_ft"),
    "dist_from_5mph_ft_delta": d("dist_from_5mph_ft"),
    "t_stop_before": before.get("t_stop"),
    "t_stop_after": after.get("t_stop"),
    "t_stop_delta": d("t_stop"),
    "peak_decel_before": before.get("peak_decel"),
    "peak_decel_after": after.get("peak_decel"),
    "peak_decel_delta": d("peak_decel"),
    "min_gap_ft_before": before.get("min_gap_ft"),
    "min_gap_ft_after": after.get("min_gap_ft"),
    "creep_after": after.get("creep"),
  }


def main(repo=None, build=None, out=None):
  kw = prepare(repo, build)
  P = kw["P"]
  personalities = (
    ("aggressive", P.aggressive),
    ("standard", P.standard),
    ("relaxed", P.relaxed),
  )
  cases = [
    ("stopped_lead_25", "stopped_lead", 25.0, 0.0, 35.0, False),
    ("stopped_lead_5", "stopped_lead", 5.0, 0.0, 20.0, False),
    ("stopping_lead_25", "stopping_lead", 25.0, -1.5, 35.0, False),
    ("stopping_lead_5", "stopping_lead", 5.0, -1.5, 20.0, False),
    ("no_lead_25", "no_lead", 25.0, 0.0, 40.0, False),
    ("no_lead_5", "no_lead", 5.0, 0.0, 40.0, False),
    ("emergency_25", "emergency", 25.0, 0.0, 15.0, True),
  ]
  result = {
    "note": (
      "devc4-test3 chauffeur final stop only. Before = official 1.0 "
      "m/s²/s longcontrol ramp, no taper, final_stop_mpc=False. After = "
      "FinalStopTaper + stopping_accel_step + stopping-lead tau / taper "
      "comp. T_FOLLOW Agg remains 1.25. No hard min-gap floor. Plant "
      "tau 0.25. Jerks exclude first 0.15 s and one-frame >4 spikes. "
      "Not road-tested."
    ),
    "research": {
      "tsb": "Toyota T-SB-0059-23 (NHTSA MC-10244133, Aug 2023): 2020-2022 "
             "Highlander Hybrid drivers may feel a momentary regen-to-hydraulic "
             "transition (downhill / light braking). Brake-actuator ECU flash.",
      "patent": "Toyota US2016/0039292: regen → cooperative → friction as "
                "speed falls; ECU holds deceleration constant across the handoff.",
      "tt": "Tomorrow's Technician: Toyota systems engage conventional brakes "
            "between about 7-17 mph (older Prius below 7 mph); panic uses "
            "friction immediately and suspends regen.",
      "openpilot": "Official sunnypilot/dev longcontrol ramps to stopAccel "
                   "(-2.0) at 1.0 m/s²/s while still rolling, starting when "
                   "should_stop (v < 0.3 m/s ≈ 0.67 mph). Toyota hybrid "
                   "actuator delay is 0.05 s. stopAccel is -2.0.",
    },
    "personalities": {},
  }
  for pname, p in personalities:
    print(f"=== {pname} ===", file=sys.stderr)
    bucket = {}
    for key, kind, mph, alead, sim_t, em in cases:
      print(f"  {key} before...", file=sys.stderr)
      b = sim_stop(p, False, kind, mph, kw, a_lead=alead, sim_t=sim_t, emergency=em)
      print(f"  {key} after...", file=sys.stderr)
      a = sim_stop(p, True, kind, mph, kw, a_lead=alead, sim_t=sim_t, emergency=em)
      c = compare(b, a)
      bucket[key] = {"before": b, "after": a, "compare": c}
      print(
        f"    last1 plant jerk {c['jerk_plant_last1mph_before']} → "
        f"{c['jerk_plant_last1mph_after']}  hold cmd "
        f"{c['hold_jerk_cmd_before']} → {c['hold_jerk_cmd_after']}  "
        f"a@1mph {c['a_at_1mph_before']} → {c['a_at_1mph_after']}  "
        f"a@0 {c['a_at_0_before']} → {c['a_at_0_after']}  "
        f"gap {c['final_gap_ft_before']} → {c['final_gap_ft_after']}  "
        f"t_stop {c['t_stop_before']} → {c['t_stop_after']}  "
        f"peak_decel {c['peak_decel_before']} → {c['peak_decel_after']}",
        file=sys.stderr,
      )
    result["personalities"][pname] = bucket

  dest = Path(out) if out else Path("/workspace/handoff/devc4-test3-dev_finalstop_sim.json")
  dest.write_text(json.dumps(result, indent=2, default=float))
  print(f"wrote {dest}")
  return result


if __name__ == "__main__":
  repo = sys.argv[1] if len(sys.argv) > 1 else None
  build = sys.argv[2] if len(sys.argv) > 2 else None
  out = sys.argv[3] if len(sys.argv) > 3 else None
  main(repo, build, out)
