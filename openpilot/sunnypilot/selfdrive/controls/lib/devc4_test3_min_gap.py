"""devc4-test3: hard minimum-gap floor at every speed.

The planned standstill bumper gap — Aggressive/Standard 24.69 ft
(STOP_DISTANCE 6.0 m + 5 ft), Relaxed 29.69 ft (6.0 m + 10 ft) — is a
hard floor at every speed, not only at a stop. The car must never
plan or command a closer gap; if a cut-in or a transient puts it
inside the floor, it brakes to restore and never commands positive
accel until the gap is back at or above the floor.

Used as:
  * An MPC danger-constraint floor (never *plan* through it).
  * A post-plan safety check applied after launch take-up and the
    final-stop taper, so those hooks cannot override it.
  * The launch-module and Aggressive-boost gate (together with the
    desired follow gap T*v + STOP + blended stop-extra): take-up and
    the +0.4 boost must not close the gap below desired.

Cut-ins that appear already inside the floor cannot be restored
instantly: radar/vision delay, the 0.05 s Toyota actuator delay, and
the 0.25 s host-plant tau mean some close-in continues until braking
takes effect. Urgent rates (FCW, short TTC) pass through at full
braking. Not road-tested. panda and docs/SAFETY.md are untouched.
"""
from __future__ import annotations

import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  STOP_DISTANCE, LEAD_STOP_V_FULL, LEAD_STOP_V_NONE, LEAD_DANGER_FACTOR,
  COMFORT_BRAKE, get_T_FOLLOW, get_lead_stop_extra, get_safe_obstacle_distance,
  get_stopped_equivalence_factor,
)

FT = 0.3048

# Comfortable restore when inside the floor and not an emergency.
# The first frame is at least this; then 0.50 m/s³ toward a PD target.
# FCW / TTC < 2 s use the raw plan if it is more urgent.
FLOOR_RESTORE_A = 0.40          # m/s²; minimum comfortable restore brake
FLOOR_RESTORE_KP = 0.40         # 1/s²; on meters under the floor
FLOOR_RESTORE_KD = 0.80         # 1/s; on closing speed
FLOOR_RESTORE_JERK = 0.50       # m/s³; comfortable restore onset
FLOOR_URGENT_TTC = 2.0          # s
FLOOR_HYST_M = 0.10             # m; release once gap >= floor (tiny chatter band)


def _finite(*xs) -> bool:
  return all(x is not None and np.isfinite(x) for x in xs)


def get_min_gap_floor(personality, experimental_mode=False) -> float:
  """Hard bumper-gap floor in meters: the planned standstill gap,
  applied at every speed. Agg/Std 7.524 m (24.69 ft) in Chill;
  Relaxed 9.048 m (29.69 ft). Aggressive experimental has no +5 ft
  extra, so the floor is STOP_DISTANCE only — unused on this
  always-Chill branch."""
  return float(STOP_DISTANCE + get_lead_stop_extra(personality, experimental_mode))


def blended_stop_extra(v_lead, personality, experimental_mode=False) -> float:
  """Lead-speed blend of the personality stop extra. Full at a stopped
  lead, zero at or above ~11 mph, same breakpoints as the MPC."""
  extra = get_lead_stop_extra(personality, experimental_mode)
  if not _finite(v_lead):
    return 0.0
  return float(np.interp(float(v_lead), [LEAD_STOP_V_FULL, LEAD_STOP_V_NONE],
                         [extra, 0.0]))


def desired_bumper_gap(v_ego, v_lead, personality, experimental_mode=False) -> float:
  """Equal-speed desired bumper gap, meters: T_FOLLOW*v + STOP + blended
  stop-extra. The MPC v²/(2*COMFORT_BRAKE) terms cancel when v_ego ≈
  v_lead, so this is what the comfort cost actually tracks."""
  v = float(v_ego) if _finite(v_ego) else 0.0
  v = max(v, 0.0)
  T = get_T_FOLLOW(personality, experimental_mode, v)
  return float(T * v + STOP_DISTANCE + blended_stop_extra(v_lead, personality,
                                                          experimental_mode))


def danger_factor_for_floor(v_ego, v_lead, personality, experimental_mode=False) -> float:
  """LEAD_DANGER_FACTOR, raised if needed so the slack danger
  constraint never allows a planned bumper gap below the floor.

  At standstill the official 0.75 * STOP would allow ~6.0 ft of
  planned close-in past the standstill gap; this returns 1.0 so the
  danger bound sits on the floor. At city and highway speeds 0.75 *
  desired is already well above the floor, so the official factor is
  unchanged.
  """
  floor = get_min_gap_floor(personality, experimental_mode)
  extra = blended_stop_extra(v_lead, personality, experimental_mode)
  v_e = max(float(v_ego), 0.0) if _finite(v_ego) else 0.0
  v_l = max(float(v_lead), 0.0) if _finite(v_lead) else 0.0
  T = get_T_FOLLOW(personality, experimental_mode, v_e)
  desired = get_safe_obstacle_distance(v_e, T)
  if desired <= 1e-6:
    return 1.0
  # Constraint is (x_obstacle - x_ego) >= factor * desired, and
  # x_obstacle - x_ego = gap + v_lead²/2a - extra. Want gap >= floor.
  need = (floor + get_stopped_equivalence_factor(v_l) - extra) / desired
  return float(max(LEAD_DANGER_FACTOR, min(need, 1.0)))


def _jerk_toward(prev, target, jerk, dt) -> float:
  if target > prev:
    return min(target, prev + jerk * dt)
  if target < prev:
    return max(target, prev - jerk * dt)
  return float(target)


def apply_min_gap_floor(a_raw, should_stop, d_rel, v_ego, v_lead=None,
                        a_prev=None, floor=None, dt=0.05, fcw=False,
                        accel_min=-3.5, v_ego_stopping=0.25):
  """Post-plan safety: if bumper gap is under the floor, never command
  positive accel, and brake to restore. Returns (a, should_stop, intervened).

  Comfortable PD restore with a jerk-limited onset; FCW or TTC under
  2 s takes the more-urgent of the raw plan and the restore target.
  Passes through bit-identically when there is no lead, the gap is
  at or above the floor, or inputs are non-finite.
  """
  if not _finite(floor) or floor <= 0.0 or not _finite(d_rel):
    return a_raw, should_stop, False
  if d_rel >= floor:
    return a_raw, should_stop, False

  v_e = float(v_ego) if _finite(v_ego) else 0.0
  v_l = float(v_lead) if _finite(v_lead) else 0.0
  v_rel = v_e - v_l  # >0 = closing
  under = float(floor) - float(d_rel)

  a_restore = -FLOOR_RESTORE_A - FLOOR_RESTORE_KP * under - FLOOR_RESTORE_KD * max(v_rel, 0.0)
  a_restore = float(np.clip(a_restore, accel_min, 0.0))

  # Already inside the floor and still closing: take the restore
  # target immediately. FCW or a short TTC does the same.
  urgent = bool(fcw)
  if v_rel > 0.05:
    urgent = True
    if _finite(d_rel) and (float(d_rel) / v_rel) < FLOOR_URGENT_TTC:
      urgent = True

  if urgent:
    a = a_restore
    if _finite(a_raw) and a_raw < a:
      a = float(a_raw)
  else:
    prev = float(a_prev) if _finite(a_prev) else (float(a_raw) if _finite(a_raw) else 0.0)
    # Instantly drop a positive request (that's lifting, not a snap
    # onto the brakes), then jerk-limit the restore onset.
    if prev > 0.0:
      prev = 0.0
    dtc = dt if (_finite(dt) and dt > 0.0) else 0.05
    a = _jerk_toward(prev, a_restore, FLOOR_RESTORE_JERK, dtc)
    if _finite(a_raw) and a_raw < a:
      a = float(a_raw)
  # Hard rule: never positive inside the floor.
  if a > 0.0:
    a = 0.0
  if not _finite(a_raw):
    # Keep a non-finite raw (solver fault) from becoming a number we
    # would rather not invent; still zero if we somehow got here.
    a = min(a, 0.0)

  ss = should_stop
  if v_e < v_ego_stopping:
    ss = True
  return float(a), bool(ss), True
