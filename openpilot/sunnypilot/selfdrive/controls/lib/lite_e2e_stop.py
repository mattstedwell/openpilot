"""dev-c4-sunnypilot-lite: stop ~10 ft before the driving model's planned stop.

Matt: "Add 10 ft stopping distance to all lead cars and to stop lights and
stop signs." The lead part lives in long_mpc (LEAD_STOP_EXTRA). This is the
light / stop-sign part, for the model's e2e candidate (the only planner
input that sees lights and signs).

Reuses the idea of devc4 641d4220b (15 ft before the model's planned stop,
skipped when a nearby lead explains the stop), rewritten for this tree's
modelV2.action candidate:
  * A planned stop is the first point in the model plan where speed is below
    PLANNED_STOP_V and the plan then stays (nearly) there for STAY_S.
  * If a radar lead is within LEAD_EXPLAINS_M of that point, nothing changes
    (the lead MPC already keeps 10 ft more behind the lead).
  * Otherwise the e2e accel is lowered to the constant decel that stops at
    (planned stop - EXTRA), clipped at MAX_EXTRA_DECEL: if that would need
    harder braking, the car brakes at MAX_EXTRA_DECEL and stops somewhere
    between the target and the model's own point. The model's own braking
    is never weakened.
  * Once the target is reached (or ego is stopped and the model still plans
    to stay stopped), shouldStop is held so it does not creep up to the
    model's line. When the model no longer plans to stay stopped (green /
    go), this returns the stock values unchanged.
It can only lower accel or set shouldStop; it never adds acceleration.
Not road-tested.
"""
from __future__ import annotations

import numpy as np

ENABLED = True
EXTRA = 10 * 0.3048 + 0.30   # m (+0.30 m margin for the smooth finish: sim stops then land at >= base, ~11 ft)
PLANNED_STOP_V = 0.5         # m/s
STAY_S = 2.0                 # s the plan must stay stopped
STAY_DX = 1.0                # m allowed movement while "stopped"
LEAD_EXPLAINS_M = 5.0        # m
MAX_EXTRA_DECEL = 2.0        # m/s^2: never brake harder than this for the extra 10 ft
HOLD_X = 0.8                 # m: at/inside this, hold the stop
MOVING_V = 0.15              # m/s
HOLD_ROLL_V = 0.3            # m/s: force the hold only below this (stock should_stop speed)
# Smooth finish (Matt: full stops "too jerky"): the old constant-decel profile
# reached the stop still at ~-1.7..-2.0 m/s^2 (a hard grab right at the line).
# Now the reference profile is constant decel TAPER_A1 = TAPER_C + TAPER_K *
# TAPER_V down to TAPER_V, then a speed-proportional finish -(TAPER_C + TAPER_K * v)
# that ends near -0.15 m/s^2 (continuous at TAPER_V). The request is that
# profile's decel at the current speed scaled by (distance the profile needs) /
# (distance left): harder if the room is short, softer if there is spare room,
# so the stop point does not move forward. (With TAPER_K = 0 this is exactly the
# old -v^2 / 2d.) Still only ever lowers accel.
SMOOTH = True
TAPER_V = 2.5                # m/s (~5.6 mph)
TAPER_C = 0.15               # m/s^2 at the stop
TAPER_K = 0.4                # 1/s (-1.15 at 2.5 m/s, -0.55 at 1 m/s, -0.27 at 0.3 m/s)


def planned_stop_x(model) -> float | None:
  try:
    x = np.asarray(model.position.x, dtype=float)
    v = np.asarray(model.velocity.x, dtype=float)
    t = np.asarray(model.position.t, dtype=float)
  except Exception:
    return None
  n = min(len(x), len(v), len(t))
  if n < 2:
    return None
  x, v, t = x[:n], v[:n], t[:n]
  for i in np.flatnonzero(v < PLANNED_STOP_V):
    if t[i] + STAY_S > t[-1]:
      # not enough horizon left to confirm; require the plan to end stopped
      if v[-1] < PLANNED_STOP_V and x[-1] - x[i] < STAY_DX:
        return float(x[i])
      return None
    x_later = float(np.interp(t[i] + STAY_S, t, x))
    if x_later - x[i] < STAY_DX:
      return float(x[i])
  return None


def taper_dist(v: float) -> float:
  """Distance (m) to stop from v <= TAPER_V under a = -(TAPER_C + TAPER_K * v)."""
  v = max(float(v), 0.0)
  ct = TAPER_C / TAPER_K
  return (v - ct * np.log1p(v / ct)) / TAPER_K


def profile(v: float) -> tuple[float, float]:
  """(decel > 0, distance to stop) of the smooth reference profile at speed v."""
  v = max(float(v), 0.0)
  if v <= TAPER_V:
    return TAPER_C + TAPER_K * v, taper_dist(v)
  a1 = TAPER_C + TAPER_K * TAPER_V
  return a1, taper_dist(TAPER_V) + (v * v - TAPER_V * TAPER_V) / (2.0 * a1)


def stop_accel(v_ego: float, dist: float) -> float:
  """Braking (<= 0) to stop smoothly within dist (m)."""
  v = max(float(v_ego), 0.0)
  if not SMOOTH:
    return max(-(v * v) / (2.0 * max(dist, 0.4)), -MAX_EXTRA_DECEL)
  dec, need = profile(v)
  a = -dec * need / max(float(dist), 0.05)
  return max(a, -MAX_EXTRA_DECEL)


class E2eStopState:
  """Per-planner memory: the last stop target, carried forward by ego travel."""
  def __init__(self):
    self.prev_target = None

  def reset(self):
    self.prev_target = None


def shift_e2e_stop(model, lead, v_ego: float, a_e2e: float, should_stop_e2e: bool,
                   state: E2eStopState | None = None, dt: float = 0.05) -> tuple[float, bool]:
  a_e2e = float(a_e2e)
  should_stop_e2e = bool(should_stop_e2e)
  if not ENABLED:
    if state is not None:
      state.reset()
    return a_e2e, should_stop_e2e
  x_stop = planned_stop_x(model)
  if x_stop is None or (lead is not None and lead.present and float(lead.dRel) < x_stop + LEAD_EXPLAINS_M):
    if state is not None:
      state.reset()
    return a_e2e, should_stop_e2e
  v_ego = max(float(v_ego), 0.0)
  target_x = x_stop - EXTRA
  if SMOOTH and state is not None:
    # The model replans from the (now slower) ego state, so its "planned stop"
    # (first v < PLANNED_STOP_V point) creeps toward the car as the smooth
    # profile slows it earlier than the model would; without memory the target
    # collapsed and the car stopped ~1 m before the 10 ft point. Keep the earlier
    # target (moved by ego travel) when the new one is closer, never beyond the
    # model's own current stop point. Only the extra 10 ft offset is affected:
    # the model's own braking stays in the min() below.
    if state.prev_target is not None:
      target_x = max(target_x, min(state.prev_target - v_ego * dt, x_stop))
    state.prev_target = target_x
  if target_x <= HOLD_X:
    if v_ego > MOVING_V:
      a_req = stop_accel(v_ego, max(target_x, 0.4))
    else:
      a_req = 0.0
    if SMOOTH and v_ego > HOLD_ROLL_V:
      # still rolling (above stock should_stop's 0.3 m/s): keep braking smoothly
      # instead of forcing Stopping early (which grabbed and stopped ~1 m short).
      return min(a_e2e, a_req), should_stop_e2e
    return min(a_e2e, a_req), True
  a_req = stop_accel(v_ego, target_x)
  return min(a_e2e, a_req), should_stop_e2e
