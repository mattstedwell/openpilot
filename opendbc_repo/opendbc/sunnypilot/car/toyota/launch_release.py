"""dev-c4-sunnypilot-lite: Toyota ACCEL_CMD brake-hold release at launch
(reused from devc4-test3 t309u / dev-c4-human-launch).

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
