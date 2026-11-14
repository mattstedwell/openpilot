"""devc4-test3: smooth, gentler positive accel ceilings (t314u).

Replaces the piecewise-linear C3X caps (1.6/1.2/0.8/0.6 at 0/10/25/40 m/s,
Relaxed 80%, Aggressive +0.4 tapering out by 22 mph, plus the t312u
40-45 mph linear blend) with one monotone, C1-continuous curve per
personality: a monotone cubic Hermite (PCHIP, Fritsch-Carlson) through a
few knots, so there are no kinks. Shape:

  * a bit gentler than C3X everywhere (older passenger comfort);
  * falls with speed roughly like a fixed fraction of power (the
    30%-PWR intent): ~1/v at highway speed;
  * Aggressive and Standard blend onto Relaxed's curve over 35-50 mph
    with a smootherstep weight (C2, no step or kink; t316u widened it
    from 40-45 mph) and equal it at >= 50 mph;
  * Aggressive's low-speed extra is part of its own curve (no boost
    flag switching, so no step when it used to arm or disarm).

Hills: the Toyota PCM's ACCEL_CMD is a vehicle-accel request and the PCM
adds the grade torque itself, so holding speed uphill only needs ~0
accel; these ceilings never go below 0.4 m/s² and do not affect it.

These are only ceilings on *positive* accel. Braking is unchanged and
the planner still never exceeds the MPC lead-follow plan (min of MPC and
cruise, then clipped to this ceiling). Nothing road-tested.
"""
from __future__ import annotations

import math

MPH = 0.44704

# Knots: speed (mph) -> ceiling (m/s²). t317u leaned all of them ~10%
# lower ("less acceleration is always better"). Highway (>= 50 mph) values are
# Relaxed's and shared by all personalities.
_KNOTS_MPH = (0., 10., 20., 30., 40., 50., 60., 70., 80., 90.)
_RELAXED = (1.05, 0.97, 0.84, 0.72, 0.62, 0.56, 0.51, 0.47, 0.44, 0.42)
# Agg/Std knots above 40 mph only shape their own curve inside the 35-50
# mph blend (they keep falling gently; the blend does the merging).
_STANDARD = (1.25, 1.15, 0.99, 0.84, 0.71, 0.63, 0.56, 0.51, 0.47, 0.45)
_AGGRESSIVE = (1.55, 1.36, 1.12, 0.92, 0.77, 0.68, 0.60, 0.54, 0.50, 0.47)

# t316u: wide, C2 (smootherstep) blend of Agg/Std onto Relaxed's curve,
# 35 -> 50 mph (was linear 40 -> 45 mph in t312u).
HWY_BLEND_START = 35. * MPH  # m/s
HWY_BLEND_FULL = 50. * MPH   # m/s

# Planner output rise limit on positive accel (m/s³), by speed.
JERK_UP_BP = (0., 20.)       # m/s
JERK_UP_V = (0.6, 0.45)      # m/s³ (t317u: leaned gentler)


def _pchip_slopes(x, y):
  """Fritsch-Carlson monotone slopes."""
  n = len(x)
  h = [x[i + 1] - x[i] for i in range(n - 1)]
  d = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
  m = [0.0] * n
  # three-point end slopes, clipped to stay monotone (same as scipy's
  # PchipInterpolator, which is not a device dependency)
  def edge(h0, h1, d0, d1):
    e = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if e * d0 <= 0.0:
      return 0.0
    if d0 * d1 <= 0.0 and abs(e) > 3 * abs(d0):
      return 3 * d0
    return e
  m[0] = edge(h[0], h[1], d[0], d[1])
  m[-1] = edge(h[-1], h[-2], d[-1], d[-2])
  for i in range(1, n - 1):
    if d[i - 1] * d[i] <= 0.0:
      m[i] = 0.0
    else:
      w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
      m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
  return m


class _Pchip:
  def __init__(self, x, y):
    self.x = [float(v) for v in x]
    self.y = [float(v) for v in y]
    self.m = _pchip_slopes(self.x, self.y)

  def __call__(self, v):
    x, y, m = self.x, self.y, self.m
    if v <= x[0]:
      return y[0]
    if v >= x[-1]:
      return y[-1]
    i = 0
    while x[i + 1] < v:
      i += 1
    h = x[i + 1] - x[i]
    t = (v - x[i]) / h
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]


_X = [k * MPH for k in _KNOTS_MPH]
_CURVES = {"relaxed": _Pchip(_X, _RELAXED), "standard": _Pchip(_X, _STANDARD),
           "aggressive": _Pchip(_X, _AGGRESSIVE)}


def _smootherstep(x0, x1, v):
  """0 -> 1 with zero 1st and 2nd derivatives at both ends (C2)."""
  if v <= x0:
    return 0.0
  if v >= x1:
    return 1.0
  t = (v - x0) / (x1 - x0)
  return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def smooth_accel_cap(v_ego, personality_name: str) -> float:
  """Positive accel ceiling (m/s²) for 'aggressive' / 'standard' /
  'relaxed'. Agg/Std blend onto Relaxed over 35-50 mph (C2 weight)."""
  v = float(v_ego)
  v = max(v, 0.0) if math.isfinite(v) else 0.0
  rel = _CURVES["relaxed"](v)
  if personality_name == "relaxed":
    return rel
  own = _CURVES["aggressive" if personality_name == "aggressive" else "standard"](v)
  w = _smootherstep(HWY_BLEND_START, HWY_BLEND_FULL, v)
  return (1.0 - w) * own + w * rel


def jerk_up(v_ego) -> float:
  v = float(v_ego)
  v = max(v, 0.0) if math.isfinite(v) else 0.0
  if v >= JERK_UP_BP[1]:
    return JERK_UP_V[1]
  return JERK_UP_V[0] + (JERK_UP_V[1] - JERK_UP_V[0]) * v / JERK_UP_BP[1]


def limit_accel_rise(a_target, a_prev, v_ego, dt) -> float:
  """Rate-limit only *increases* of positive accel. Decreases (less
  accel, more braking) pass through unchanged; from a braking command
  the first step above 0 is also limited from 0, not from a_prev."""
  a = float(a_target)
  ceiling = max(float(a_prev), 0.0) + jerk_up(v_ego) * float(dt)
  return min(a, ceiling) if a > 0.0 else a
