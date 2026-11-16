"""sunnypilot-lite: open the follow gap a little faster after a launch from a stop.

Matt (Nov 15 2026): after accelerating from a stop behind a lead, the gap sags
below the personality follow distance. Stock MPC does the same: ego
overshoots the lead's speed when the lead finishes accelerating.

A later request (same day) raised the ease-off from about 1 mph under the lead
to about 2 mph (V_OPEN = 2.0 * MPH). The accel floor stays -0.3 m/s^2.

A follow-up requested a ramp-in during the launch (originally 2.5 to 6.5 mph,
then 10 to 14 mph). With lead lock now holding to 10 mph and handing off by
11.5 mph, gap-open must not start until that handoff completes, otherwise
the ease-off would fight the lock. Below RAMP_START_V (11.5 mph) the
module behaves exactly like no gap-open (normal launch, lead lock, and
boost). The ease-off episode is not started, and the cap is not applied,
until ego reaches RAMP_START_V. From there the under-lead offset ramps
with a C2 smootherstep to the full V_OPEN at RAMP_END_V (15.5 mph, a 4 mph
window as before). Standstill behind a lead still marks the car eligible,
so a later 11.5 mph crossing can start the episode; eligibility alone
does not change accel.

The offset itself is also proportional to how short the gap is, so it stays
human-like: up to ~2 mph under the lead when the gap is well short of the
personality follow distance (at or below SHORT_FRAC of T_FOLLOW*v +
STOP_DISTANCE), shrinking with a C2 smootherstep to 0 as the gap approaches
100% of that target. The cap turns off exactly at 100% (no lingering under
the lead) but the launch episode stays armed, so a later sag can ease off
again — for example a second pull-away before the 35 mph / 25 s expiry.

The ease-off itself is kept gentle: the cap's own decel is limited to
A_SOFT (about -0.10 m/s^2), well above the hard A_FLOOR of -0.3, so a
large catch-up error does not slam into the floor. As the gap fills from
SHORT_FRAC to 100%, that cap is also blended back toward the incoming
plan with the same C2 smootherstep, so the release is gradual rather than
an abrupt lift. There is no fixed 2 mph hold after the gap is closed.

The cap is min()'d with the incoming plan, including while launch_boost and
lead_lock are still active. Those hooks still run first (boost may raise the
accel floor; lock holds the stopped gap to 10 mph and keeps its safety
overrides). Gap-open can only reduce accel, never weaken braking, and it
drops immediately if the lead is slowing, stopped, missing, or FCW is set,
so a stopped or slowing lead is never ignored.

Constraints (all enforced here and in the planner hook):
  * gas ease-off only: the cap is min()'d with the incoming plan, so it never
    weakens braking. The cap's own command is limited to A_SOFT (~-0.10 m/s^2);
    A_FLOOR (-0.3 m/s^2) remains only a hard bound.
  * below RAMP_START_V (11.5 mph, after the lead-lock handoff) the launch
    is not touched, so take-up, t_move, lead lock and the first part of
    boost stay stock. The ease-off episode is not started before 11.5 mph.
  * eligibility is remembered from a standstill behind a lead, but the
    episode itself starts only at 11.5 mph; it expires at END_V (~35 mph) or
    MAX_EPISODE_S, so steady moving follow is unchanged. Reaching 100% of
    the target turns the cap off; it does not end the episode.
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
RAMP_START_V = 11.5 * MPH    # m/s: start after lead-lock handoff completes
RAMP_END_V = 15.5 * MPH      # m/s: full speed-ramp from here (4 mph window)
END_V = 35.0 * MPH           # m/s: episode ends (highway / leaving the launch)
MAX_EPISODE_S = 25.0         # s after the episode starts (at 11.5 mph)
SHORT_FRAC = 0.90            # full V_OPEN at or below this fraction of the target gap
V_OPEN = 2.0 * MPH           # m/s (~2 mph): max under-lead offset when the gap is well short
TAU = 1.2                    # s: speed error -> accel
A_SOFT = -0.10               # m/s^2: max ease-off the cap itself may command
A_FLOOR = -0.30              # m/s^2: hard bound (the cap uses A_SOFT, not this)
A_LEAD_MIN = -0.30           # m/s^2: lead not braking
JERK_DOWN = 1.2              # m/s^3: ease-off rate (comfort)
JERK_UP = 1.0                # m/s^3: release rate
MISSING_MAX_S = 1.0          # s of radar dropout before the episode ends


def _smootherstep01(x: float) -> float:
  x = min(max(x, 0.0), 1.0)
  return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def v_open_scale(v_ego: float) -> float:
  """0 at or below RAMP_START_V, 1 at or above RAMP_END_V, C2 smootherstep between."""
  span = RAMP_END_V - RAMP_START_V
  if span <= 1e-6:
    return 1.0 if float(v_ego) >= RAMP_END_V else 0.0
  return _smootherstep01((float(v_ego) - RAMP_START_V) / span)


def gap_short_scale(gap: float, desired: float) -> float:
  """1 when the gap is well short of target (SHORT_FRAC), 0 at 100% of target.

  C2 smootherstep between. This is the human-like 'ease off more when close,
  sit under less as the gap fills in' term.
  """
  if desired <= 1e-3:
    return 0.0
  lo = SHORT_FRAC * desired
  hi = desired
  g = float(gap)
  if g <= lo:
    return 1.0
  if g >= hi:
    return 0.0
  return 1.0 - _smootherstep01((g - lo) / (hi - lo))


def v_open_offset(v_ego: float, gap: float | None = None, desired: float | None = None) -> float:
  """Ramped, gap-proportional under-lead offset (m/s).

  Speed ramp: 0 until 11.5 mph, full scale by 15.5 mph. If gap and desired are
  given, also shrinks to 0 as the gap reaches 100% of the follow target.
  """
  scale = v_open_scale(v_ego)
  if gap is not None and desired is not None:
    scale *= gap_short_scale(gap, desired)
  return V_OPEN * scale


def desired_follow_gap(v_ego: float, personality) -> float:
  """Personality follow distance: T_FOLLOW * v_ego + STOP_DISTANCE (no stop extra)."""
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc as LM
  return float(LM.get_T_FOLLOW(personality) * max(float(v_ego), 0.0) + LM.STOP_DISTANCE)


class LiteGapOpen:
  def __init__(self):
    self.reset()

  def reset(self):
    self.armed = False
    self.episode = False
    self.active = False
    self.held = False  # cap has been applied this episode; used to jerk-release at 100%
    self.motion_t = 0.0
    self.missing = 0.0
    self.applied = 0.0  # last cap we applied (for jerk limiting)

  def _end_episode(self, a_target: float) -> tuple[bool, float]:
    self.armed = False
    self.episode = False
    self.active = False
    self.held = False
    self.applied = a_target
    return False, a_target

  def _jerk_limit(self, desired_out: float, a_target: float, a_prev: float, dt: float) -> float:
    if desired_out < a_prev:
      out = max(desired_out, a_prev - JERK_DOWN * dt)
    else:
      out = min(desired_out, a_prev + JERK_UP * dt)
    out = min(out, a_target)
    # the cap itself never eases off harder than A_SOFT; if the plan is already
    # below that, the early-return has already passed the plan through
    return max(out, min(a_target, A_SOFT))

  def update(self, lead, v_ego: float, standstill: bool, fcw: bool, personality, a_target: float,
             a_prev: float, dt: float, lock_active: bool = False, boost_active: bool = False) -> tuple[bool, float]:
    """Returns (active, a_target). a_target is never raised, and braking is never weakened.

    lock_active / boost_active are accepted for planner compatibility. The cap is a
    min() on top of those hooks and no longer waits for them to hand off; they do
    not by themselves disable the cap. A stopped or slowing lead still drops it.
    """
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

    # Remember eligibility from a standstill behind a lead, but do not start
    # the ease-off episode (or apply any cap) before 11.5 mph (after the
    # lead-lock handoff). Below that, accel is exactly the incoming plan
    # (launch / lock / boost are untouched).
    if standstill or v_ego < ARM_V:
      if present:
        self.armed = True
        self.episode = False
        self.held = False
        self.motion_t = 0.0
        self.applied = a_target
      else:
        self.armed = False
        self.episode = False
        self.held = False
        self.motion_t = 0.0
      return False, a_target

    # still below the 11.5 mph start: eligible, but the episode has not armed
    if v_ego < RAMP_START_V:
      self.applied = a_target
      return False, a_target

    if self.armed and not self.episode:
      self.episode = True
      self.held = False
      self.motion_t = 0.0
    if not self.episode:
      return False, a_target

    self.motion_t += dt
    # expire the launch episode: leaving the speed/time window, or a long dropout
    if v_ego >= END_V or self.motion_t >= MAX_EPISODE_S or self.missing > MISSING_MAX_S:
      return self._end_episode(a_target)

    # below 11.5 mph is already returned above; v_scale is 0 until RAMP_START_V
    v_scale = v_open_scale(v_ego)
    if v_scale <= 1e-3 or not present or fcw:
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
    g_scale = gap_short_scale(d, desired)
    offset = V_OPEN * v_scale * g_scale

    # target reached: jerk-limit the release and turn the cap off so we do not
    # linger under the lead. The episode stays armed so a later sag (a second
    # pull-away, a catch-up after the lead pauses) can ease off again.
    if g_scale <= 1e-3 or d >= desired:
      if self.held and a_prev < a_target - 1e-6:
        out = self._jerk_limit(a_target, a_target, a_prev, dt)
        if out < a_target - 1e-6:
          self.active = True
          self.applied = out
          return True, out
      self.held = False
      self.applied = a_target
      return False, a_target

    # already opening at least the current (ramped, gap-scaled) offset: don't
    # hold the car back while the lead is still accelerating and we are slower
    v_err = (v_lead - offset) - v_ego
    if offset <= 1e-4 or v_err >= 0.0:
      self.applied = a_target
      return False, a_target

    a_cap = v_err / TAU
    a_cap = max(a_cap, A_SOFT)
    # gradual fade-out: blend the cap back toward the incoming plan as the
    # gap fills from SHORT_FRAC to 100%, so release is smooth, not abrupt
    a_cap = a_target + g_scale * (a_cap - a_target)
    # never raise the plan (min on top of launch_boost / lead_lock)
    desired_out = min(a_target, a_cap)
    # plan already at or below the cap (including already braking): do not touch it
    if desired_out >= a_target - 1e-6:
      self.applied = a_target
      return False, a_target

    out = self._jerk_limit(desired_out, a_target, a_prev, dt)
    if out >= a_target - 1e-6:
      self.applied = a_target
      return False, a_target
    self.active = True
    self.held = True
    self.applied = out
    return True, out
