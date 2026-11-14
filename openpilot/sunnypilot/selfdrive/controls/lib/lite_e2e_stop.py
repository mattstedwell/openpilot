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
EXTRA = 10 * 0.3048          # m
PLANNED_STOP_V = 0.5         # m/s
STAY_S = 2.0                 # s the plan must stay stopped
STAY_DX = 1.0                # m allowed movement while "stopped"
LEAD_EXPLAINS_M = 5.0        # m
MAX_EXTRA_DECEL = 2.0        # m/s^2: never brake harder than this for the extra 10 ft
HOLD_X = 0.8                 # m: at/inside this, hold the stop
MOVING_V = 0.15              # m/s


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


def shift_e2e_stop(model, lead, v_ego: float, a_e2e: float, should_stop_e2e: bool) -> tuple[float, bool]:
  a_e2e = float(a_e2e)
  should_stop_e2e = bool(should_stop_e2e)
  if not ENABLED:
    return a_e2e, should_stop_e2e
  x_stop = planned_stop_x(model)
  if x_stop is None:
    return a_e2e, should_stop_e2e
  if lead is not None and lead.present and float(lead.dRel) < x_stop + LEAD_EXPLAINS_M:
    return a_e2e, should_stop_e2e
  v_ego = max(float(v_ego), 0.0)
  target_x = x_stop - EXTRA
  if target_x <= HOLD_X:
    if v_ego > MOVING_V:
      a_req = -(v_ego * v_ego) / (2.0 * max(target_x, 0.4))
      a_req = max(a_req, -MAX_EXTRA_DECEL)
    else:
      a_req = 0.0
    return min(a_e2e, a_req), True
  a_req = max(-(v_ego * v_ego) / (2.0 * target_x), -MAX_EXTRA_DECEL)
  return min(a_e2e, a_req), should_stop_e2e
