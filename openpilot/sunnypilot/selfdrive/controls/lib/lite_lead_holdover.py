"""dev-c4-sunnypilot-lite: brief lead holdover (lost-lead fix).

Ported from dev-c4-human devc4_human_city.LeadHoldover (96ca223ce1).
When radarState drops leadOne for up to HOLD_S, the lead MPC keeps a
phantom lead instead of the official "fake fast lead 50 m ahead":
  * it starts from the last real dRel / vLead,
  * it decelerates at HOLD_DECEL_SCALE x its last (lagged) aLeadK when that
    was braking (capped at HOLD_DECEL_MAX) and never accelerates,
  * modelProb = 0.0, so it can never raise FCW by itself (long_mpc only
    counts crash_cnt when modelProb > 0.9).
No hold when the last lead was farther than HOLD_D_MAX or pulling away
faster than ego by more than HOLD_PULLAWAY. It only ever adds an obstacle,
so it can only make the plan brake earlier, never accelerate.
Not road-tested.
"""
from __future__ import annotations

import math

ENABLED = True
HOLD_S = 1.0
HOLD_D_MAX = 40.0          # m
HOLD_PULLAWAY = 1.0        # m/s
HOLD_DECEL_SCALE = 2.0
HOLD_DECEL_MAX = -4.0      # m/s^2


def _f(x) -> float:
  try:
    return float(x)
  except (TypeError, ValueError):
    return float("nan")


def lead_present(lead) -> bool:
  return lead is not None and bool(getattr(lead, "present", False))


class HeldLead:
  __slots__ = ("present", "status", "dRel", "vLead", "vLeadK", "aLeadK", "aLeadTau", "modelProb", "vRel", "yRel", "held")

  def __init__(self, d, v, a, tau, v_ego):
    self.present = self.status = True
    self.dRel, self.vLead, self.vLeadK, self.aLeadK, self.aLeadTau = d, v, v, a, tau
    self.modelProb = 0.0   # never raises FCW by itself
    self.vRel = v - v_ego
    self.yRel = 0.0
    self.held = True


class LeadHoldover:
  def __init__(self):
    self.reset()

  def reset(self):
    self.last = None
    self.t_held = math.inf
    self.d = self.v = self.a = self.tau = float("nan")

  def update(self, lead, v_ego, dt):
    """Real lead when present, a held lead during a short dropout, else the input."""
    if not ENABLED:
      self.reset()
      return lead
    v = _f(v_ego)
    if lead_present(lead):
      vals = tuple(_f(getattr(lead, k, float("nan"))) for k in ("dRel", "vLead", "aLeadK", "aLeadTau"))
      self.last = vals if all(math.isfinite(x) for x in vals) else None
      self.t_held = 0.0 if self.last is not None else math.inf
      if self.last is not None:
        self.d, self.v, self.tau = self.last[0], self.last[1], self.last[3]
        self.a = max(HOLD_DECEL_MAX, HOLD_DECEL_SCALE * min(self.last[2], 0.0))
      return lead
    if self.last is None or not math.isfinite(v):
      return lead
    d0, v0 = self.last[0], self.last[1]
    if d0 > HOLD_D_MAX or v0 > v + HOLD_PULLAWAY or self.t_held + dt > HOLD_S + 1e-6:  # float-safe: a full 1.0 s at 20 Hz
      self.reset()
      return lead
    self.t_held += dt
    v_new = max(0.0, self.v + self.a * dt)
    self.d = max(0.1, self.d + 0.5 * (self.v + v_new) * dt - v * dt)
    self.v = v_new
    return HeldLead(self.d, self.v, self.a if self.v > 0.0 else 0.0, self.tau, v)
