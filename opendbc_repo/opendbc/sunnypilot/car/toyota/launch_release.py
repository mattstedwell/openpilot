"""devc4-test3: Toyota ACCEL_CMD brake-hold release at launch (t309u).

At a standstill the official Stopping state holds ACCEL_CMD at
stopAccel (-2.0). When longcontrol leaves Stopping and asks for
*positive* accel, the carcontroller's 4 m/s³ windup limit spends
~0.5 s unwinding -2.0 back to zero before any drive torque, which is
most of the launch delay. This drops the previous command to a light
hold (LAUNCH_RELEASE_HOLD, still a braking command) in one step, then
the normal 4 m/s³ windup continues toward the request.

Only when: long control active, longControlState is pid (Stopping has
already been left), ego is stopped (v < 0.1 m/s), the request is
positive, and the previous command is below the light hold. It never
raises the command above the request (rate_limit still clamps to it)
and never touches braking: a request at or below zero is passed
through with the official 4 m/s³ winddown. Not road-tested.
"""
LAUNCH_RELEASE_HOLD = -0.3   # m/s²; still a brake request, never drive
LAUNCH_RELEASE_V_MAX = 0.1   # m/s; standstill only
LAUNCH_RELEASE_A_MIN = 0.0   # m/s²; request must be positive


def release_prev_accel(prev_accel, accel_request, v_ego, pid_active, long_active):
  if not (long_active and pid_active):
    return prev_accel
  if v_ego >= LAUNCH_RELEASE_V_MAX or accel_request <= LAUNCH_RELEASE_A_MIN:
    return prev_accel
  if prev_accel >= LAUNCH_RELEASE_HOLD:
    return prev_accel
  return min(LAUNCH_RELEASE_HOLD, accel_request)


# dev-c4-human launch075 (t361u, target ~0.75 s from lead move to ego move).
# While still stopped and longcontrol already asks for positive accel
# (it only does that after the planner's gated take-up: lead pulling
# away, not slowing, gap >= set stopping distance, no FCW), send at least
# LAUNCH_KICK_ACCEL so the PCM lets go of the standstill brake faster.
# It ends the moment the car moves (v >= LAUNCH_RELEASE_V_MAX) or the
# request drops to LAUNCH_KICK_REQ_MIN or below. Braking: if the request
# turns into braking at low speed, any leftover positive command is
# dropped at once (no 4 m/s³ winddown from a positive value), so the
# brake builds immediately. Not road-tested; the real PCM's own resume
# delay is not modelled anywhere.
LAUNCH_KICK_ACCEL = 0.65     # m/s²; 0 = off
LAUNCH_KICK_REQ_MIN = 0.0    # m/s²; request must be above this
LAUNCH_DROP_V_MAX = 3.0      # m/s; leftover-positive drop on a braking request


def launch_kick_floor(accel_request, v_ego, pid_active, long_active):
  """Returns the minimum ACCEL_CMD for the launch kick, or None."""
  if LAUNCH_KICK_ACCEL <= 0.0 or not (long_active and pid_active):
    return None
  if v_ego >= LAUNCH_RELEASE_V_MAX or accel_request <= LAUNCH_KICK_REQ_MIN:
    return None
  return LAUNCH_KICK_ACCEL


def drop_positive_on_brake(prev_accel, accel_request, v_ego, long_active):
  """Braking request at low speed: never wind down from a positive command."""
  if long_active and v_ego < LAUNCH_DROP_V_MAX and accel_request < 0.0 and prev_accel > 0.0:
    return 0.0
  return prev_accel


# launch075 (Matt: firm, fast response to the lead beats smoothness): below
# LOW_SPEED_WINDDOWN_V the ACCEL_CMD may fall (more braking) at up to
# LOW_SPEED_WINDDOWN_RATE instead of the official 4 m/s³, so the brake
# builds sooner near a stopped / slowing lead. Only the braking direction
# changes; windup (more accel) keeps the official limit. Bounds are still
# the panda / carcontroller -3.5..2.0 clip.
LOW_SPEED_WINDDOWN_RATE = 20.0  # m/s³ (4.0 = official)
LOW_SPEED_WINDDOWN_V = 3.0      # m/s


def winddown_limit(v_ego, default_limit, dt):
  """Per-frame (negative) rate limit for decreasing ACCEL_CMD."""
  if v_ego < LOW_SPEED_WINDDOWN_V and LOW_SPEED_WINDDOWN_RATE > 4.0:
    return min(default_limit, -LOW_SPEED_WINDDOWN_RATE * dt)
  return default_limit
