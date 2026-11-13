#!/usr/bin/env python3
"""Closed-loop host sims for t301u/t303u/t304u: remaining stop jerk,
already-stopped 24.7 ft (Agg/Std), earlier slowing-lead braking, and
Relaxed-as-reference.

Before = 5ea5397a5c: FinalStopTaper (no personality), final_stop_mpc
on, no LeadSlow, official Toyota PCM (4.0 + frozen I). After =
LeadSlow + personality-gated taper (Relaxed is 5ea539-identical
approach) + post-PID 0.30 PCM comfort at stopping / last mph.

Plant: 0.05 s Toyota actuator delay, tau 0.25, pre-PID 4.0 rate
limit, TSS2-ish PID with freeze-on-stopping, optional 7-17 mph
regen-friction bump when the command is changing, hold latch at
v=0 if the command steps.

Import-safe. Nothing road-tested.

usage: test3_final_stop2_sim.py [REPO] [BUILD_DIR] [out.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

MPH = 0.44704
FT = 0.3048
LAST_1MPH = 1.0 * MPH
STOP_ACCEL = -2.0
BLEND_LO_MPH = 7.0
BLEND_HI_MPH = 17.0


def worktree_root() -> Path:
  here = Path(__file__).resolve()
  return here.parents[5]


def _plant_jerk_sample(j, ev, t, ignore_first=0.15):
  if t <= ignore_first:
    return False
  if j > 4.0:
    return False
  return True


class ToyotaPlant:
  """2021 Highlander Hybrid TSS2-ish plant for felt jerk."""

  def __init__(self, after=True):
    self.after = after
    self.delay = [0.0] * 5  # 0.05 s at 0.01
    self.ea = 0.0
    self.prev_accel = 0.0  # official pre-PID rate-limit state
    self.last_sent = 0.0
    self.i = 0.0
    self.aego = 0.0
    self.tick = 0
    self.blend_bump = 0.0
    self.hold_bump = 0.0
    self.latched = False
    self.tau = 0.25
    self.last_pcm = 0.0

  def step_pcm(self, actuators_accel, ev, stopping, dt=0.01):
    """Official 4.0 rate limit + PID, optional comfort, every 3 frames."""
    import numpy as np
    from opendbc.sunnypilot.car.toyota.smooth_stop_hold import (
      smooth_pcm_accel, fade_stopping_integral, COMFORT_JERK,
    )
    self.tick += 1
    # Toyota long runs every 3 ctrl frames.
    if self.tick % 3 != 1 and self.tick != 1:
      return self.last_sent

    dt_tick = dt * 3.0
    # Official pre-PID rate limit 4.0 m/s³
    step4 = 4.0 * dt_tick
    cmd = float(np.clip(actuators_accel,
                        self.prev_accel - step4,
                        self.prev_accel + step4))
    self.prev_accel = cmd

    # Simplified TSS2 PID: kp=0, kf=1, ki 0.5 at 2 m/s to 0.25 at 5.
    ki = float(np.interp(ev, [2.0, 5.0], [0.5, 0.25]))
    # a_ego_future ≈ plant ea (already delayed/filtered)
    error = cmd - self.ea
    if not stopping:
      self.i = self.i + error * ki * dt_tick
      self.i = float(np.clip(self.i, -3.5 - cmd, 2.0 - cmd))
    # freeze I when stopping; official also slowly unwinds at 0.03
    if stopping:
      self.i -= 0.03 * dt_tick * (1.0 if self.i > 0 else (-1.0 if self.i < 0 else 0.0))
    pcm = cmd + self.i  # kf=1, kp=0
    pcm = float(np.clip(pcm, -3.5, 2.0))

    if self.after:
      pcm = smooth_pcm_accel(pcm, actuators_accel, self.last_sent,
                             dt_tick, stopping, ev, True)
      if stopping:
        # fade leftover I at comfort rate (mirrors carcontroller)
        class _P:
          pass
        p = _P()
        p.i = self.i
        fade_stopping_integral(p, dt_tick)
        self.i = p.i

    self.last_sent = pcm
    return pcm

  def step(self, actuators_accel, ev, stopping, dt=0.01):
    import numpy as np
    pcm = self.step_pcm(actuators_accel, ev, stopping, dt)
    # 0.05 s delay
    self.delay.append(pcm)
    delayed = self.delay.pop(0)
    # First-order actuator
    self.ea = self.ea + (delayed - self.ea) * (1.0 - np.exp(-dt / self.tau))
    # Regen-friction blend bump 7-17 mph if the command is changing
    v_mph = ev / MPH if MPH else 0.0
    dcmd = abs(pcm - self.last_pcm) / dt if dt > 0 else 0.0
    if BLEND_LO_MPH < v_mph < BLEND_HI_MPH and dcmd > 0.20:
      target_bump = 0.12 * (1.0 if (pcm - self.last_pcm) < 0 else -1.0)
      # Onset at ~2 m/s³ so the disturbance itself is not a 12 m/s³ spike
      db = max(-2.0 * dt, min(2.0 * dt, target_bump - self.blend_bump))
      self.blend_bump += db
    else:
      self.blend_bump *= max(0.0, 1.0 - 3.0 * dt)  # ~0.33 s decay
    # Hold latch at v=0 if the command steps (PCM pressure bump)
    if ev <= 0.02 and not self.latched:
      self.latched = True
      step = abs(pcm - self.last_pcm)
      if step > 0.04:
        self.hold_bump = -min(0.30, step)
    # Ease the latch at 0.30 so we measure the command's jerk, not a
    # numerical hold spike.
    if self.hold_bump != 0.0:
      db = max(-0.30 * dt, min(0.30 * dt, 0.0 - self.hold_bump))
      self.hold_bump += db
    self.last_pcm = pcm
    felt = self.ea + self.blend_bump + self.hold_bump
    return pcm, felt


def prepare(repo=None, build=None):
  from openpilot.sunnypilot.selfdrive.controls.tests.test3_launchfix_sim import (
    prepare as _prep,
  )
  return _prep(repo, build)


def sim_case(personality, after, kind, v0_mph, kw, a_lead=-1.0, sim_t=30.0,
             v_lead_end=None, emergency=False, v_lead0_mph=None):
  """kind:
    'stopping_lead' — both at v0, lead brakes at a_lead from t=0.5 to 0
    'stopped_lead'  — lead already at rest, ego from v0
    'no_lead'       — cruise to 0
    'emergency'     — tight gap, lead stopped
    'slowing_lead'  — both at v0 (or v_lead0), lead brakes at a_lead from
                      t=0.5 until v_lead_end (None = to a stop)
    'highway'       — I-10 style: both at cruise follow, lead slows to
                      v_lead_end and holds
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
  from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_lead_slow import (
    LeadSlow,
  )

  dt_mdl, dt_ctrl = 0.05, 0.01
  is_relaxed = personality == P.relaxed
  is_agg = personality == P.aggressive
  v0 = v0_mph * MPH
  T = get_T_FOLLOW(personality, False, v0)
  extra = get_lead_stop_extra(personality, False)
  target = STOP_DISTANCE + extra
  present = True
  t_brake = 0.5
  lv_end = 0.0 if v_lead_end is None else v_lead_end * MPH
  if kind == "stopping_lead":
    d0 = T * v0 + target
    lv0, ev0 = v0, v0
    lv_end = 0.0
  elif kind == "stopped_lead":
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
    d0 = 8.0 if v0_mph >= 15 else 4.0
    lv0, ev0 = 0.0, v0
    a_lead = 0.0
    t_brake = 0.0
    emergency = True
  elif kind in ("slowing_lead", "highway"):
    lv0 = (v_lead0_mph * MPH) if v_lead0_mph is not None else v0
    ev0 = v0
    d0 = desired_follow_gap(ev0, lv0, personality, False)
  else:
    raise ValueError(kind)

  mpc = LongitudinalMpc()
  mpc.final_stop_mpc = True  # both: 5ea539 already has this
  mpc.lead_slow_mpc = bool(after)
  ls = LeadLaunch()
  boost_sm = ChillAggressiveLaunch()
  taper = FinalStopTaper()
  slow = LeadSlow() if after else None
  plant = ToyotaPlant(after=after)
  v_cruise = ev0 if kind in ("highway",) else (
    0.0 if kind == "no_lead" else max(v0 + 5.0, 25.0)
  )
  if kind == "highway":
    # Stay in cruise follow; set-speed is our current / a bit above
    v_cruise = max(v0, lv_end + 2.0 * MPH)

  ex, ev, ea = 0.0, ev0, 0.0
  lx, lv = d0, lv0
  plan_a, v_des, last, state, ss = 0.0, ev0, 0.0, "pid", False
  a_cruise_st = plan_a
  prev_plan = 0.0
  prev_out = 0.0
  prev_felt = 0.0
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
  a_at_8mph = a_at_5mph = a_at_1mph = a_at_0 = None
  final_gap = None
  desired0 = desired_follow_gap(ev0, lv0, personality, False) if present else None
  min_gap_vs_desired = None
  pcm_cmd = 0.0

  for k in range(int(sim_t / dt_ctrl)):
    t = k * dt_ctrl
    if kind in ("stopping_lead", "slowing_lead", "highway"):
      if t >= t_brake and lv > lv_end + 1e-6:
        la = a_lead
      else:
        la = 0.0
    else:
      la = 0.0
    lv = max(lv + la * dt_ctrl, 0.0)
    if kind in ("slowing_lead", "highway") and lv < lv_end:
      lv = lv_end
    if kind == "stopping_lead" and lv < 0:
      lv = 0.0
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
      if after and slow is not None:
        a_t = slow.apply(
          a_t, ev, rs.leadOne if present else None, personality, False,
          dt_mdl, desired_gap=desired, a_prev=plan_a, experimental_mode=False)
      # FinalStopTaper is on both sides (5ea539). After passes personality
      # so Agg/Std get the 24.7 / short-taper path; Relaxed stays 5ea539.
      if after:
        a_t = taper.apply(
          a_t, ev, rs.leadOne if present else None, False, dt_mdl,
          should_stop=bool(ss), a_prev=plan_a, personality=personality,
          experimental_mode=False, desired_gap=desired)
      else:
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

    if state == "pid" and ss and ev < 0.3:
      state = "stopping"
    elif state == "stopping" and (not ss) and ev > 0.3:
      state = "pid"

    if state == "stopping":
      out = last
      if out > STOP_ACCEL:
        standstill = ev <= 0.02
        out = stopping_accel_step(out, standstill, dt_ctrl, ev, stop_accel=STOP_ACCEL)
        out = max(min(out, 0.0), STOP_ACCEL)
    else:
      out = plan_a

    pcm, felt = plant.step(out, ev, state == "stopping", dt_ctrl)

    if (not saw_standstill) and ev <= 0.02 and t > 0.1 and out <= 0.0:
      saw_standstill = True
      t_standstill = t
      standstill_step = out - prev_out
      standstill_jerk = abs(standstill_step) / dt_ctrl

    # Toyota ACCEL_CMD updates every 3 frames (0.03 s). Measuring
    # |dpcm|/0.01 on a held-then-step command triple-counts jerk.
    if k > 0 and pcm != prev_out:
      j_out = abs(pcm - prev_out) / (dt_ctrl * 3.0)
    else:
      j_out = 0.0
    if _plant_jerk_sample(j_out, ev, t):
      peak_j_out = max(peak_j_out, j_out)
      if 0.02 < ev <= LAST_1MPH:
        peak_j_out_last1 = max(peak_j_out_last1, j_out)
      if saw_standstill and t_standstill is not None and t_standstill < t <= t_standstill + 2.0:
        hold_peak_j_out = max(hold_peak_j_out, j_out)

    last = out
    ev_prev = ev
    ev = max(0.0, ev + felt * dt_ctrl)
    if ev <= 0 and felt < 0:
      ev = 0.0
    ex += ev * dt_ctrl

    j_plant = abs(felt - prev_felt) / dt_ctrl if k > 0 else 0.0
    if _plant_jerk_sample(j_plant, ev, t):
      peak_j_plant = max(peak_j_plant, j_plant)
      if 0.02 < ev <= LAST_1MPH:
        peak_j_plant_last1 = max(peak_j_plant_last1, j_plant)
      if saw_standstill and t_standstill is not None and t_standstill < t <= t_standstill + 2.0:
        hold_peak_j_plant = max(hold_peak_j_plant, j_plant)
    prev_felt = felt
    prev_out = pcm
    pcm_cmd = pcm

    peak_decel = min(peak_decel, felt, out)
    if 0.02 < ev <= LAST_1MPH:
      peak_decel_last1 = min(peak_decel_last1, felt)

    if present:
      gap = lx - ex
      min_gap = min(min_gap, gap)
      final_gap = gap
      if desired0 is not None:
        dv = gap - desired_follow_gap(ev, lv, personality, False)
        if min_gap_vs_desired is None or dv < min_gap_vs_desired:
          min_gap_vs_desired = dv
    else:
      final_gap = None
      min_gap = None

    if t_first_brake is None and t >= t_brake and plan_a < -0.10:
      t_first_brake = t
    if t_5mph is None and ev_prev > 5 * MPH >= ev:
      t_5mph = t
      x_at_5 = ex
      a_at_5mph = felt
    if t_1mph is None and ev_prev > LAST_1MPH >= ev:
      t_1mph = t
      a_at_1mph = felt
    if a_at_8mph is None and ev_prev > 8 * MPH >= ev:
      a_at_8mph = felt
    if t_stop is None and ev <= 0.02 and ev_prev > 0.02:
      t_stop = t
      a_at_0 = felt
      if x_at_5 is not None:
        dist_from_5 = ex - x_at_5
    # End conditions
    if kind in ("highway",) and t > 20.0 and abs(ev - lv) < 0.3 and t > t_brake + 8.0:
      break
    if t_stop is not None and ev <= 0.02 and t > t_stop + 3.0:
      break

  return {
    "kind": kind,
    "v0_mph": v0_mph,
    "v_lead_end_mph": None if v_lead_end is None else v_lead_end,
    "a_lead": a_lead,
    "after": after,
    "personality": int(personality),
    "emergency": emergency,
    "T": round(float(T), 3),
    "planned_standstill_ft": round(target / FT, 2),
    "desired0_ft": None if desired0 is None else round(float(desired0) / FT, 2),
    "d0_ft": round(d0 / FT, 1),
    "t_first_brake": None if t_first_brake is None else round(t_first_brake, 2),
    "t_5mph": None if t_5mph is None else round(t_5mph, 2),
    "t_1mph": None if t_1mph is None else round(t_1mph, 2),
    "t_stop": None if t_stop is None else round(t_stop, 2),
    "t_standstill": None if t_standstill is None else round(t_standstill, 2),
    "dist_from_5mph_m": None if dist_from_5 is None else round(float(dist_from_5), 3),
    "dist_from_5mph_ft": None if dist_from_5 is None else round(float(dist_from_5) / FT, 2),
    "final_gap_ft": None if final_gap is None else round(float(final_gap) / FT, 2),
    "min_gap_ft": None if min_gap is None else round(float(min_gap) / FT, 2),
    "min_gap_vs_desired_m": None if min_gap_vs_desired is None else round(float(min_gap_vs_desired), 3),
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
    "hold_peak_jerk_cmd": round(float(hold_peak_j_out), 3),
    "hold_peak_jerk_plant": round(float(hold_peak_j_plant), 3),
    "a_at_8mph": None if a_at_8mph is None else round(float(a_at_8mph), 3),
    "a_at_5mph": None if a_at_5mph is None else round(float(a_at_5mph), 3),
    "a_at_1mph": None if a_at_1mph is None else round(float(a_at_1mph), 3),
    "a_at_0": None if a_at_0 is None else round(float(a_at_0), 3),
    "rest_decel_design": FINAL_STOP_REST_DECEL,
    "design_jerk": FINAL_STOP_JERK,
    "final_v_mph": round(float(ev) / MPH, 2),
    "final_v_lead_mph": round(float(lv) / MPH, 2),
    "stopped": t_stop is not None,
  }


def compare(before, after):
  def d(key):
    b, a = before.get(key), after.get(key)
    if b is None or a is None:
      return None
    return round(float(a) - float(b), 3)

  keys = (
    "t_first_brake", "peak_decel", "peak_jerk_plant", "peak_jerk_cmd",
    "peak_jerk_plant_last1mph", "peak_jerk_cmd_last1mph",
    "hold_peak_jerk_cmd", "hold_peak_jerk_plant", "standstill_jerk",
    "a_at_1mph", "a_at_0", "final_gap_ft", "min_gap_ft",
    "min_gap_vs_desired_m", "t_stop", "dist_from_5mph_ft",
  )
  out = {}
  for k in keys:
    out[f"{k}_before"] = before.get(k)
    out[f"{k}_after"] = after.get(k)
    out[f"{k}_delta"] = d(k)
  return out


def vs_relaxed(agg_or_std, relaxed):
  def d(key):
    b, a = relaxed.get(key), agg_or_std.get(key)
    if b is None or a is None:
      return None
    return round(float(a) - float(b), 3)
  keys = (
    "t_first_brake", "peak_decel", "peak_jerk_plant", "peak_jerk_cmd",
    "min_gap_ft", "final_gap_ft", "t_stop", "min_gap_vs_desired_m",
    "peak_jerk_plant_last1mph", "hold_peak_jerk_cmd", "a_at_0",
  )
  out = {"relaxed_T": relaxed.get("T"), "this_T": agg_or_std.get("T"),
         "relaxed_desired0_ft": relaxed.get("desired0_ft"),
         "this_desired0_ft": agg_or_std.get("desired0_ft"),
         "relaxed_planned_ft": relaxed.get("planned_standstill_ft"),
         "this_planned_ft": agg_or_std.get("planned_standstill_ft")}
  for k in keys:
    out[f"{k}_this"] = agg_or_std.get(k)
    out[f"{k}_relaxed"] = relaxed.get(k)
    out[f"{k}_delta"] = d(k)
  return out


def main(repo=None, build=None, out=None):
  kw = prepare(repo, build)
  P = kw["P"]
  personalities = (
    ("aggressive", P.aggressive),
    ("standard", P.standard),
    ("relaxed", P.relaxed),
  )
  # (key, kind, v0, a_lead, sim_t, v_lead_end, emergency)
  cases = [
    ("stopped_lead_25", "stopped_lead", 25.0, 0.0, 35.0, None, False),
    ("stopped_lead_5", "stopped_lead", 5.0, 0.0, 25.0, None, False),
    ("stopping_lead_25", "stopping_lead", 25.0, -1.5, 35.0, 0.0, False),
    ("stopping_lead_5", "stopping_lead", 5.0, -1.5, 25.0, 0.0, False),
    ("no_lead_25", "no_lead", 25.0, 0.0, 45.0, None, False),
    ("no_lead_5", "no_lead", 5.0, 0.0, 45.0, None, False),
    ("emergency_25", "emergency", 25.0, 0.0, 15.0, None, True),
    # t303u slowing leads
    ("slow_m10_45_to_stop", "slowing_lead", 45.0, -1.0, 40.0, 0.0, False),
    ("slow_m20_45_to_stop", "slowing_lead", 45.0, -2.0, 30.0, 0.0, False),
    ("slow_m10_45_to_20", "slowing_lead", 45.0, -1.0, 25.0, 20.0, False),
    ("slow_m20_45_to_20", "slowing_lead", 45.0, -2.0, 20.0, 20.0, False),
    # I-10 style
    ("i10_m06_70_to_55", "highway", 70.0, -0.6, 25.0, 55.0, False),
    ("i10_m10_70_to_55", "highway", 70.0, -1.0, 25.0, 55.0, False),
  ]
  result = {
    "note": (
      "devc4-test3 t301u/t303u/t304u. Before = 5ea5397a5c (FinalStopTaper "
      "without personality, final_stop_mpc on, no LeadSlow, official Toyota "
      "PCM 4.0+frozen I). After = LeadSlow + personality-gated taper "
      "(Relaxed approach is 5ea539-identical) + post-PID 0.30 PCM comfort "
      "at stopping / last mph only. T_FOLLOW Agg remains 1.25. No hard "
      "min-gap floor. Plant: 0.05 s delay, tau 0.25, TSS2-ish PID, 7-17 "
      "mph blend bump if command is changing, hold latch if command steps "
      "at v=0. Jerks exclude first 0.15 s and one-frame >4 spikes. "
      "Not road-tested."
    ),
    "research": {
      "tsb": "Toyota T-SB-0059-23 (NHTSA MC-10244133, Aug 2023)",
      "patent": "Toyota US2016/0039292: hold deceleration across regen→friction",
      "tt": "Tomorrow's Technician: conventional brakes ~7-17 mph",
      "pcm": "What reaches the car is pcm_accel_cmd after the Toyota PID; "
             "frozen I at pid→stopping is a step on ACCEL_CMD. Comfort "
             "path drops I and rate-limits at 0.30 only in stopping / last mph.",
    },
    "personalities": {},
    "vs_relaxed": {},
  }
  after_runs = {}
  for pname, p in personalities:
    print(f"=== {pname} ===", file=sys.stderr)
    bucket = {}
    for key, kind, mph, alead, sim_t, vend, em in cases:
      print(f"  {key} before...", file=sys.stderr, flush=True)
      b = sim_case(p, False, kind, mph, kw, a_lead=alead, sim_t=sim_t,
                   v_lead_end=vend, emergency=em)
      print(f"  {key} after...", file=sys.stderr, flush=True)
      a = sim_case(p, True, kind, mph, kw, a_lead=alead, sim_t=sim_t,
                   v_lead_end=vend, emergency=em)
      c = compare(b, a)
      bucket[key] = {"before": b, "after": a, "compare": c}
      after_runs[(pname, key)] = a
      print(
        f"    t_brake {c['t_first_brake_before']}→{c['t_first_brake_after']}  "
        f"peak_decel {c['peak_decel_before']}→{c['peak_decel_after']}  "
        f"j_plant {c['peak_jerk_plant_before']}→{c['peak_jerk_plant_after']}  "
        f"j_last1 {c['peak_jerk_plant_last1mph_before']}→{c['peak_jerk_plant_last1mph_after']}  "
        f"hold {c['hold_peak_jerk_cmd_before']}→{c['hold_peak_jerk_cmd_after']}  "
        f"gap {c['final_gap_ft_before']}→{c['final_gap_ft_after']}  "
        f"min_gap {c['min_gap_ft_before']}→{c['min_gap_ft_after']}  "
        f"t_stop {c['t_stop_before']}→{c['t_stop_after']}",
        file=sys.stderr, flush=True,
      )
    result["personalities"][pname] = bucket

  # How Agg/Std after compare to Relaxed after (the reference)
  for key, *_ in cases:
    rel = after_runs.get(("relaxed", key))
    if rel is None:
      continue
    block = {}
    for pname in ("aggressive", "standard"):
      this = after_runs.get((pname, key))
      if this is None:
        continue
      block[pname] = vs_relaxed(this, rel)
    result["vs_relaxed"][key] = block
    # brief stderr
    ag = block.get("aggressive", {})
    print(
      f"  vs_relaxed {key}: Agg t_brake {ag.get('t_first_brake_this')} vs Rel "
      f"{ag.get('t_first_brake_relaxed')}  peak {ag.get('peak_decel_this')} vs "
      f"{ag.get('peak_decel_relaxed')}  jerk {ag.get('peak_jerk_plant_this')} vs "
      f"{ag.get('peak_jerk_plant_relaxed')}  min_gap {ag.get('min_gap_ft_this')} vs "
      f"{ag.get('min_gap_ft_relaxed')}  final {ag.get('final_gap_ft_this')} vs "
      f"{ag.get('final_gap_ft_relaxed')}",
      file=sys.stderr, flush=True,
    )

  dest = Path(out) if out else Path("/workspace/handoff/devc4-test3-dev_finalstop2_sim.json")
  dest.write_text(json.dumps(result, indent=2, default=float))
  print(f"wrote {dest}")
  return result


if __name__ == "__main__":
  repo = sys.argv[1] if len(sys.argv) > 1 else None
  build = sys.argv[2] if len(sys.argv) > 2 else None
  out = sys.argv[3] if len(sys.argv) > 3 else None
  main(repo, build, out)
