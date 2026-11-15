"""sunnypilot-lite: vision + radar agreement helpers for the launch paths.

Matt: "Use vision + radar where you think it's appropriate." radard already
fuses the two for leadOne (a radar track matched to the model's lead when the
model's lead prob > 0.5, a vision-only lead with radar=False when no track
matches, and the low-speed radar override), and lite_lead_holdover covers a
short total dropout. This module only adds the asymmetric rule for the
things that make the car GO sooner or harder:

  * launch / boost need a radar-measured lead (leadOne.radar True). A
    vision-only lead never shortens the delay or boosts the launch, so a
    vision false positive cannot cause a launch.
  * if the model also sees a lead (prob > MODEL_PROB_MIN), it must agree the
    lead is moving (v >= MODEL_V_MOVING); if the model has no lead, radar
    alone is enough.
  * (evaluated, OFF: USE_MODEL_GAP) using the smaller of radar dRel and the
    model's lead distance for the gap gate; it regressed creep-cycle gaps.
Stopping and holding are untouched here: they keep using leadOne (radar or
vision, whichever radard reports) plus the holdover, so a vision dropout can
never remove a stop. Not road-tested.
"""
from __future__ import annotations

import math

REQUIRE_RADAR = True
# sunnypilot-lite-test-fix (Matt, Nov 14 road test): vision is OUT of the launch
# decision. A radar-measured lead that pulls away is enough; the model's lead
# speed (which lags at a launch) no longer has to agree. Kept as a flag only.
REQUIRE_MODEL_AGREE = False
# Off: sim showed the model's (lagged, camera-offset) lead distance blocking
# quick launches in creep cycles, after which the stock launch ended up to
# 13 cm closer to the lead than base. Kept for evaluation only.
USE_MODEL_GAP = False
MODEL_PROB_MIN = 0.5
MODEL_V_MOVING = 0.15     # m/s: model agrees the lead is rolling
RADAR_TO_CAMERA = 1.52    # m, as radard


def _first(seq, default=None):
  try:
    return float(seq[0]) if len(seq) > 0 else default
  except Exception:
    return default


def model_lead_seen(model_lead) -> bool:
  if model_lead is None:
    return False
  try:
    return float(model_lead.prob) > MODEL_PROB_MIN
  except Exception:
    return False


def lead_moving_agreed(lead, model_lead) -> bool:
  """Radar-measured lead, and the model either has no lead or agrees it moves."""
  if lead is None or not lead.present:
    return False
  if REQUIRE_RADAR and not bool(getattr(lead, "radar", False)):
    return False
  if not REQUIRE_MODEL_AGREE or not model_lead_seen(model_lead):
    return True
  v = _first(model_lead.v)
  return v is not None and math.isfinite(v) and v >= MODEL_V_MOVING


def conservative_gap(lead, model_lead) -> float:
  d = float(lead.dRel)
  if USE_MODEL_GAP and model_lead_seen(model_lead):
    x = _first(model_lead.x)
    if x is not None and math.isfinite(x):
      d = min(d, x - RADAR_TO_CAMERA)
  return d


# --- driving-model stop vs. radar launch (sunnypilot-lite-test-fix) ---------
# Root cause of "car does not move until I press the gas": behind a stopped
# lead the driving model's e2e candidate keeps shouldStop and ~0 accel until
# the MODEL sees the lead move and plans to go, and lite_e2e_stop clamps the
# e2e accel to <= 0 at a standstill; the quick launch was blocked by that
# shouldStop and capped at that accel. So the launch waited on the model (its
# lag, or forever if it stays in "stop"). The radar launch now overrides the
# model's stop EXCEPT when the model is clearly stopping for something other
# than the lead: a planned stop more than LEAD_EXPLAINS_M short of the radar
# lead, once the lead has already moved LINE_CONFIRM_M away from where it sat
# (a stop line / sign / red light stays put while the lead leaves; a stop
# caused by the lead moves with it). Once that happens the launch is disarmed
# until the next standstill and the stock min (incl. the model) rules.
LEAD_EXPLAINS_M = 5.0      # m (same as lite_e2e_stop)
LINE_CONFIRM_M = 1.0       # m the lead must have moved before a short model stop counts as a line


def model_stop_blocks(model_stop_x, d_rel: float, lead_moved: float) -> bool:
  """True if the model plans a stop that the departing lead does not explain."""
  if model_stop_x is None:
    return False
  try:
    x = float(model_stop_x); d = float(d_rel); m = float(lead_moved)
  except Exception:
    return True
  if not (math.isfinite(x) and math.isfinite(d)):
    return True
  if x + LEAD_EXPLAINS_M >= d:
    return False  # the lead explains it (the model stops behind the lead)
  return bool(math.isfinite(m) and m >= LINE_CONFIRM_M)
