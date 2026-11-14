"""devc4-test3: comfortable acceleration in curves (t311u).

Matt: "I don't want to accelerate in curves such that I feel
uncomfortable." On this tree nothing limits *positive* accel in a curve:
official limit_accel_in_turns is gone, Always-Chill drops the model's
desiredAcceleration, and Smart Cruise Control Vision/Map default off.

This caps only the positive accel ceiling (the planner `cap`) with a
friction-circle style comfort budget:

  a_long_max(lat) = min(sqrt(A_TOTAL_MAX² - lat²),
                        interp(lat, [LAT_TAPER_START, LAT_ZERO], [A_TOTAL_MAX, 0]))

so total accel stays under ~1.6 m/s² and longitudinal accel tapers to 0
as lateral accel reaches ~1.4 m/s² (t317u; was 1.8 / 1.6). `lat` is the larger of the current
lateral accel (v² · controlsState.curvature) and the lateral accel the
model's planned path predicts over the next few seconds, scaled to
ego's speed (so it does not accelerate *into* a curve or mid-curve).

The ceiling falls at up to CAP_FALL_JERK and recovers at CAP_RISE_JERK
on curve exit (low jerk). It is never below 0, and the planner applies
it only as the upper clip of the plan, so braking and lead-follow
braking are never limited. Nothing road-tested.
"""
from __future__ import annotations

import math

import numpy as np

A_TOTAL_MAX = 1.6        # m/s²; friction-circle budget (long + lat); t317u: was 1.8
LAT_TAPER_START = 0.5    # m/s²; below this lateral accel, only the circle applies; t317u: was 0.6
LAT_ZERO = 1.4           # m/s²; longitudinal accel ceiling reaches 0 here; t317u: was 1.6
LOOKAHEAD_T = 3.0        # s; how far along the model plan to look for a curve
MIN_PLAN_V = 1.0         # m/s; guards curvature = yaw_rate / v at low speed
CAP_FALL_JERK = 1.0      # m/s³; how fast the ceiling may drop (curve ahead); t317u: was 1.5
CAP_RISE_JERK = 0.4      # m/s³; how fast the ceiling recovers (curve exit)


def long_accel_allowed(lat_accel: float) -> float:
  """Comfort ceiling for positive longitudinal accel at a lateral accel."""
  lat = abs(float(lat_accel)) if math.isfinite(float(lat_accel)) else 0.0
  circle = math.sqrt(max(A_TOTAL_MAX ** 2 - lat ** 2, 0.0))
  taper = float(np.interp(lat, [LAT_TAPER_START, LAT_ZERO], [A_TOTAL_MAX, 0.0]))
  return max(0.0, min(circle, taper))


def predicted_lat_accel(v_ego, yaw_rates, plan_v, t_idxs, lookahead_t=LOOKAHEAD_T) -> float:
  """Max lateral accel over the next lookahead_t seconds of the model
  plan, at ego's speed if ego is faster than the plan's speed."""
  try:
    yaw = np.abs(np.asarray(yaw_rates, dtype=float))
    vp = np.asarray(plan_v, dtype=float)
    t = np.asarray(t_idxs, dtype=float)
  except (TypeError, ValueError):
    return 0.0
  n = min(len(yaw), len(vp), len(t))
  if n == 0:
    return 0.0
  yaw, vp, t = yaw[:n], vp[:n], t[:n]
  mask = (t <= lookahead_t) & np.isfinite(yaw) & np.isfinite(vp)
  if not mask.any():
    return 0.0
  v_ref = np.maximum(vp[mask], MIN_PLAN_V)
  curv = yaw[mask] / v_ref
  v_lat = np.maximum(v_ref, max(float(v_ego), 0.0))
  return float(np.max(curv * v_lat ** 2))


class CurveComfort:
  def __init__(self):
    self.cap = None
    self.lat_now = 0.0
    self.lat_pred = 0.0
    self.limit = A_TOTAL_MAX

  def reset(self):
    self.cap = None

  def update(self, cap, v_ego, curvature, yaw_rates, plan_v, t_idxs, dt) -> float:
    """Return the curve-limited positive accel ceiling (<= cap, >= 0)."""
    cap = float(cap)
    v = max(float(v_ego), 0.0) if math.isfinite(float(v_ego)) else 0.0
    k = float(curvature) if curvature is not None and math.isfinite(float(curvature)) else 0.0
    self.lat_now = v * v * abs(k)
    self.lat_pred = predicted_lat_accel(v, yaw_rates, plan_v, t_idxs)
    self.limit = long_accel_allowed(max(self.lat_now, self.lat_pred))
    target = max(0.0, min(cap, self.limit))
    if self.cap is None:
      self.cap = target
    elif target < self.cap:
      self.cap = max(target, self.cap - CAP_FALL_JERK * dt)
    else:
      self.cap = min(target, self.cap + CAP_RISE_JERK * dt)
    # Never above the personality cap, never negative.
    self.cap = max(0.0, min(self.cap, cap))
    return float(self.cap)
