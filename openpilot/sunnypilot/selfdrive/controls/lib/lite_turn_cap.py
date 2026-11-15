"""sunnypilot-lite-test2: keep the felt g low in turns (Matt, Nov 14).

Road report: on a low-speed city RIGHT turn the car accelerated fairly hard into
the corner (lead turned out of view / set speed / launch paths). Matt's spec:
use the steering angle (CarState.steeringAngleDeg -> VehicleModel curvature)
for the lateral g = v^2 * curvature, the combined g = sqrt(lat^2 + long^2), and
limit POSITIVE longitudinal accel so that:
  * combined g stays under G_TOTAL_MAX (0.15 g) while turning,
  * no positive accel at all once lat g >= LAT_G_NO_ACCEL (0.10 g),
  * if the wheel is still winding in (|angle| growing faster than
    WIND_RATE_DEG_S beyond WIND_ANGLE_DEG), hold speed (no positive accel).
At turn speeds lat accel is small, so the curvature from the WHEEL ANGLE (not
the model path or yaw rate) is what sees an intersection corner early.
Applies only while turning (lat g >= LAT_G_ON or |angle| >= WIND_ANGLE_DEG)
and below V_MAX. It runs AFTER the quick launch, launch boost and lead lock,
so none of them, nor cruise after a lead is lost mid-turn, can push accel into
the corner. Braking is never touched (always available for safety).
Not road-tested.
"""
from __future__ import annotations

import math

ENABLED = True
G = 9.81
V_MAX = 35.0 * 0.44704       # m/s: city turn speeds only
G_TOTAL_MAX = 0.15           # g: combined (lat + long) while turning
LAT_G_NO_ACCEL = 0.10        # g: no positive accel at/above this
LAT_G_ON = 0.02              # g: cap active from here
WIND_ANGLE_DEG = 15.0        # deg steering wheel
WIND_RATE_DEG_S = 10.0       # deg/s: wheel still winding in -> hold speed
# host fallback when no VehicleModel is given (2021 Highlander: steerRatio 16.0, wheelbase 2.8194 m)
STEER_RATIO = 16.0
WHEELBASE = 2.8194


def curvature_from_angle(steering_angle_deg: float, steer_ratio: float = STEER_RATIO, wheelbase: float = WHEELBASE) -> float:
  return math.tan(math.radians(float(steering_angle_deg)) / steer_ratio) / wheelbase


def lat_g(v_ego: float, curvature: float) -> float:
  return float(v_ego) ** 2 * abs(float(curvature)) / G


def accel_limit(v_ego: float, curvature: float, steering_angle_deg: float = 0.0, steering_rate_deg: float = 0.0) -> float:
  """Max positive accel (m/s^2) allowed now; math.inf when not turning."""
  try:
    v = float(v_ego); k = float(curvature); ang = float(steering_angle_deg); rate = float(steering_rate_deg)
  except Exception:
    return math.inf
  if not (math.isfinite(v) and math.isfinite(k)) or v >= V_MAX:
    return math.inf
  lg = lat_g(v, k)
  turning = lg >= LAT_G_ON or abs(ang) >= WIND_ANGLE_DEG
  if not turning:
    return math.inf
  if math.isfinite(ang) and math.isfinite(rate) and abs(ang) >= WIND_ANGLE_DEG and ang * rate > 0.0 and \
     abs(rate) >= WIND_RATE_DEG_S:
    return 0.0  # still winding in: hold speed
  if lg >= LAT_G_NO_ACCEL:
    return 0.0
  return math.sqrt(max((G_TOTAL_MAX * G) ** 2 - (lg * G) ** 2, 0.0))


def cap_accel(a_target: float, v_ego: float, curvature: float, steering_angle_deg: float = 0.0,
              steering_rate_deg: float = 0.0) -> float:
  """Lower a positive a_target to the turn limit. Never lowers braking (a <= 0 passes through)."""
  if not ENABLED:
    return a_target
  lim = accel_limit(v_ego, curvature, steering_angle_deg, steering_rate_deg)
  a = float(a_target)
  if a <= 0.0 or a <= lim:
    return a_target
  return max(lim, 0.0)
