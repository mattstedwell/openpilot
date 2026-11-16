"""sunnypilot-lite: open the follow gap a little faster after a launch from a stop.

Matt (Nov 15 2026): after accelerating from a stop behind a lead, the gap sags
below the personality follow distance (host sim: Aggressive ~45-54 ft vs the
~58 ft target at 20 mph, recovering in 11-17 s). Stock MPC does the same: ego
overshoots the lead's speed when the lead finishes accelerating.

After a standstill launch, once lead lock has handed off (6.5 mph) and the
first MIN_MOTION_S of the launch have passed, if the gap is still below
T_FOLLOW*v_ego + STOP_DISTANCE and the lead is not braking, this softly caps
ego accel so ego sits slightly under the lead's speed (V_OPEN). That keeps
the gap opening instead of shrinking, then fades out as the gap reaches
TARGET_FRAC of the target.

Constraints (all enforced here and in the planner hook):
  * gas ease-off only: the cap is min()'d with the incoming plan, so it never
    weakens braking. The cap itself is never below A_FLOOR (-0.3 m/s^2).
  * first MIN_MOTION_S of the launch, and the lead lock up to LOCK_END_V,
    are not touched (and the cap is off while the launch boost is still on).
  * the episode is only armed from a standstill behind a lead; it expires at
    END_V (~35 mph) or MAX_EPISODE_S, so steady moving follow is unchanged.
  * a slowing / braking lead, FCW, or a missing lead drops the cap immediately
    (the stock plan is used). The cap never raises accel, so it cannot launch
    into a stopped lead or inch on a roll-stop cycle.
  * jerk-limited on and off so the ease-off does not feel like a pull-back.
  * follow-time targets, stopped gap, light/sign stops, the soft final stop
    and the turn cap are not changed (the planner applies this BEFORE the
    turn cap, which stays last).

Not road-tested.
"""
from __future__ import annotations

import math

from openpilot.cereal import log

ENABLED = True
MPH = 0.44704
ARM_V = 0.05                 # m/s: arm only from a standstill
LOCK_END_V = 6.5 * MPH       # m/s: lead lock is released at this speed; do not cap before
MIN_MOTION_S = 0.9           # s: first part of the launch is unchanged
END_V = 35.0 * MPH           # m/s: episode ends (highway / leaving the launch)
MAX_EPISODE_S = 25.0         # s after first motion
TARGET_FRAC = 0.95           # fade the cap out by this fraction of T_FOLLOW*v + STOP
FADE_START_FRAC = 0.80       # full effect at or below this fraction
V_OPEN = 0.40                # m/s (~0.9 mph): sit this much under the lead so the gap opens
TAU = 1.2                    # s: speed error -> accel
A_FLOOR = -0.30              # m/s^2: never command more than a light ease-off
A_LEAD_MIN = -0.30           # m/s^2: lead not braking
JERK_DOWN = 1.2              # m/s^3: ease-off rate (comfort)
JERK_UP = 1.0                # m/s^3: release rate
MISSING_MAX_S = 1.0          # s of radar dropout before the episode ends


def _smootherstep01(x: float) -> float:
  x = min(max(x, 0.0), 1.0)
  return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def desired_follow_gap(v_ego: float, personality) -> float:
  """Personality follow distance: T_FOLLOW * v_ego + STOP_DISTANCE (no stop extra)."""
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc as LM
  return float(LM.get_T_FOLLOW(personality) * max(float(v_ego), 0.0) + LM.STOP_DISTANCE)


def gap_fade(gap: float, desired: float) -> float:
  """1 below FADE_START_FRAC of the target, 0 at TARGET_FRAC and above."""
  if desired <= 1e-3:
    return 0.0
  lo = FADE_START_FRAC * desired
  hi = TARGET_FRAC * desired
  if hi <= lo:
    return 1.0 if gap < hi else 0.0
  return 1.0 - _smootherstep01((float(gap) - lo) / (hi - lo))


class LiteGapOpen:
  def __init__(self):
    self.reset()

  def reset(self):
    self.armed = False
    self.episode = False
    self.active = False
    self.motion_t = 0.0
    self.missing = 0.0
    self.applied = 0.0  # last cap we applied (for jerk limiting)

  def update(self, lead, v_ego: float, standstill: bool, fcw: bool, personality, a_target: float,
             a_prev: float, dt: float, lock_active: bool = False, boost_active: bool = False) -> tuple[bool, float]:
    """Returns (active, a_target). a_target is never raised, and braking is never weakened."""
    self.active = False
    a_target = float(a_target)
    a_prev = float(a_prev)
    dt = max(float(dt), 1e-3)
    if not ENABLED:
      self.reset()
      return False, a_target

    v_ego = float(v_ego)
    present = lead is not None and bool(getattr(lead, "present", False))
    self.missing = 0.0 if present else self.missing + dt

    # arm only from a standstill behind a lead; start the episode on first motion
    if standstill or v_ego < ARM_V:
      if present:
        self.armed = True
        self.episode = False
        self.motion_t = 0.0
        self.applied = a_target
      else:
        self.armed = False
        self.episode = False
        self.motion_t = 0.0
      return False, a_target

    if self.armed and not self.episode:
      self.episode = True
      self.motion_t = 0.0
    if not self.episode:
      return False, a_target

    self.motion_t += dt
    # expire the launch episode: leaving the speed/time window, or a long dropout
    if v_ego >= END_V or self.motion_t >= MAX_EPISODE_S or self.missing > MISSING_MAX_S:
      self.armed = False
      self.episode = False
      return False, a_target

    # do not touch the first 0.9 s, the lead-lock region, or a still-active boost
    if (self.motion_t < MIN_MOTION_S or v_ego < LOCK_END_V or lock_active or boost_active
        or not present or fcw):
      self.applied = a_target
      return False, a_target

    try:
      d = float(lead.dRel)
      v_lead = float(lead.vLead)
      a_lead = float(lead.aLeadK)
    except Exception:
      self.applied = a_target
      return False, a_target
    if not (math.isfinite(d) and math.isfinite(v_lead) and math.isfinite(a_lead)):
      self.applied = a_target
      return False, a_target

    # lead braking / rolling backward: drop the cap, use the stock plan
    if a_lead < A_LEAD_MIN or v_lead < -0.05:
      self.applied = a_target
      return False, a_target

    desired = desired_follow_gap(v_ego, personality)
    fade = gap_fade(d, desired)
    # already opening at least V_OPEN, or already at the target: nothing to cap
    # (don't hold the car back while the lead is still accelerating and we are already slower)
    v_err = (v_lead - V_OPEN) - v_ego
    if fade <= 1e-3 or d >= TARGET_FRAC * desired or v_err >= 0.0:
      self.applied = a_target
      return False, a_target

    a_cap = v_err / TAU
    # blend toward the incoming plan as the gap approaches the target
    a_cap = a_target + fade * (a_cap - a_target)
    a_cap = max(a_cap, A_FLOOR)
    # never raise the plan
    desired_out = min(a_target, a_cap)

    # jerk-limit the ease-off and the release, but never weaken braking
    if desired_out < a_prev:
      out = max(desired_out, a_prev - JERK_DOWN * dt)
    else:
      out = min(desired_out, a_prev + JERK_UP * dt)
    out = min(out, a_target)
    if out >= a_target - 1e-6:
      self.applied = a_target
      return False, a_target
    self.active = True
    self.applied = out
    return True, out
