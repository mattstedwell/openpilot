"""devc4-test3: Relaxed-like approach braking for Aggressive and Standard.

t304u: Relaxed approach and slowdown braking is the reference and
stays as it is on 5ea5397a5c. apply() is a bit-identical passthrough
for Relaxed. Aggressive and Standard keep their own T_FOLLOW
(1.25 / 1.45) and planned standstill gaps (24.7 ft); onset timing is
scaled to those shorter gaps so the *feel* matches Relaxed — similar
jerk (0.30 m/s³), similar peak decel, earlier onset rather than
waiting then braking harder.

Responds as soon as a lead begins slowing (a_lead ≲ -0.20) or is
already stopped, including a highway slowdown that will not halt.
Never goes below the personality's desired follow / stop distance
via earlier planning (this is not a restored min-gap PD floor).
Urgent / FCW / TTC under 2 s / gap under 3.5 m / launch (positive)
pass through at full braking. Emergency deceleration is never
weakened (commands already at or below -1.5 pass through).

The MPC-side helpers (slowing_a_lead_tau, should_use_relaxed_jerk)
are also no-ops for Relaxed so its 5ea539 prediction and weights
are unchanged. Not road-tested. panda and docs/SAFETY.md are
untouched.
"""
from __future__ import annotations

import numpy as np

from openpilot.sunnypilot.selfdrive.car import chill_relaxed
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_final_stop import (
  FINAL_STOP_JERK, FINAL_STOP_MIN_GAP, FINAL_STOP_TTC, FINAL_STOP_URGENT_FULL,
  STOP_LEAD_TAU, _finite, _jerk_toward, _lead_present, _urgent,
  lead_is_stopping,
)

DT = 0.05

# Start "as soon as the lead begins slowing," above typical aLeadK noise.
SLOWING_ALEAD = -0.20
# Relaxed already-stopped 25 mph peak on 5ea539 was about -1.38.
# Agg/Std comfortable approaches aim near this; kinematics can ask
# for more, and urgent passthrough is unchanged.
RELAXED_PEAK = 1.40
# Closing-correction time constant when already at/inside desired
# (jerk-limited catch-up, not a hard floor).
CATCHUP_TAU = 1.5
# Non-urgent floor (m/s²). Urgent / FCW already passed through.
COMFORT_MIN = -2.0


def is_relaxed(personality, experimental_mode=False) -> bool:
  """True for Chill Relaxed, the t304u reference. None is treated as
  'not this overlay' by apply() only when we need a passthrough;
  callers that want the 5ea539 FinalStop path treat None as Relaxed
  identically."""
  if personality is None:
    return True
  return bool(chill_relaxed.is_chill_relaxed(experimental_mode, personality))


def lead_is_slowing(v_lead, a_lead, v_ego=None) -> bool:
  """Clearly decelerating, including a highway slowdown that will not
  come to a full halt. Also true for a lead already stopping/stopped."""
  if lead_is_stopping(v_lead, a_lead):
    return True
  if not _finite(v_lead, a_lead):
    return False
  if a_lead > SLOWING_ALEAD:
    return False
  if _finite(v_ego) and float(v_ego) < float(v_lead) - 0.5:
    # Clearly pulling away; no need to start a slowdown response.
    return False
  return True


def slowing_a_lead_tau(v_ego, a_lead, a_lead_tau, v_lead) -> float:
  """Agg/Std: persist a slowing lead's decel in the MPC prediction,
  including slowdowns that will not halt (I-10 70→55). Relaxed keeps
  stopping_a_lead_tau only (5ea539: persist only if the lead will
  halt within 15 s)."""
  if lead_is_slowing(v_lead, a_lead, v_ego) and _finite(a_lead_tau):
    return float(min(a_lead_tau, STOP_LEAD_TAU))
  return a_lead_tau


def should_use_relaxed_jerk(personality, experimental_mode, radarstate, v_ego) -> bool:
  """Agg/Std facing a slowing or stopped lead: use Relaxed's 1.3 jerk
  factor so the MPC solution itself is earlier and gentler. T_FOLLOW
  is unchanged. get_jerk_factor itself is not modified (existing
  unit tests). Relaxed always returns False (already at 1.3)."""
  if is_relaxed(personality, experimental_mode):
    return False
  if radarstate is None:
    return False
  lead = getattr(radarstate, "leadOne", None)
  if not _lead_present(lead):
    return False
  v_lead = getattr(lead, "vLead", float("nan"))
  a_lead = getattr(lead, "aLeadK", getattr(lead, "aLead", float("nan")))
  return lead_is_slowing(v_lead, a_lead, v_ego)


def _approach_target(v_ego, v_lead, a_lead, d_rel, desired, a_raw) -> float:
  """Signed accel that matches a slowing lead and cancels closing
  before the gap reaches `desired`. Scales onset to each
  personality's own desired gap (Agg/Std stay closer than Relaxed).
  """
  a_lead_f = float(a_lead) if _finite(a_lead) else 0.0
  if a_lead_f > 0.0:
    a_lead_f = 0.0
  v = max(float(v_ego), 0.0) if _finite(v_ego) else 0.0
  vl = max(float(v_lead), 0.0) if _finite(v_lead) else 0.0
  v_rel = v - vl
  desired_f = float(desired) if _finite(desired) else 0.0
  # Behind a stopped lead the finish is planned standstill (24.7 ft
  # for Agg/Std), not T*v + standstill (that's equal-speed follow).
  if vl <= 0.5:
    from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_final_stop import (
      AGG_STD_STANDSTILL_M)
    if desired_f <= 0.0:
      desired_f = AGG_STD_STANDSTILL_M
    else:
      desired_f = min(desired_f, AGG_STD_STANDSTILL_M)
  d = float(d_rel) if _finite(d_rel) else desired_f
  s = d - desired_f

  # Default: match the lead's decel as soon as it begins slowing.
  a_target = a_lead_f

  if v_rel > 0.05:
    if s > 0.5:
      # Exact constant-decel closing cancel: a_ego = a_lead - v_rel²/(2s).
      # Larger s (Agg's shorter desired at the same d_rel) → milder,
      # later onset — "scaled to their gap."
      a_target = a_lead_f - (v_rel * v_rel) / (2.0 * s)
    else:
      # Near or inside desired and still closing: match the lead,
      # plus a small jerk-limited catch-up. Not a 1 m/s² step and
      # not a PD floor. A large extra here was overshooting I-10
      # and falling behind a -1.5 stopping lead then slamming.
      extra = min(max(v_rel / 2.5, 0.0), 0.40)
      a_target = a_lead_f - extra

  # If the lead is already stopped, this reduces to a_ego = -v²/(2s)
  # (stop at desired), or a mild hold if we are already there and
  # not closing.
  if vl <= 0.5 and v_rel <= 0.05 and s > 0.5:
    a_need = (v * v) / (2.0 * s)
    a_target = min(a_target, -a_need) if a_need > 0.0 else a_target

  # Lower the peak to Relaxed-like when kinematics still work at
  # that peak. Always allowed to at least match the lead (a lead
  # at -1.5 cannot be followed at -1.40). If we need more than
  # that to hold desired, use more (up to COMFORT_MIN).
  peak = min(-RELAXED_PEAK, a_lead_f)
  if a_target < peak:
    if s > 0.5 and v_rel > 0.05:
      a_rel_avail = (-peak) + a_lead_f
      need = (v_rel * v_rel) / (2.0 * s)
      if a_rel_avail + 1e-6 >= need:
        a_target = peak
    elif s > 0.5 and vl <= 0.5:
      a_need = (v * v) / (2.0 * s)
      if a_need <= (-peak):
        a_target = peak
    # else keep a_target (needed)

  if a_target < COMFORT_MIN:
    a_target = COMFORT_MIN
  if a_target > 0.0:
    a_target = 0.0
  return float(a_target)


class LeadSlow:
  """Planner-side earlier, gentler approach for Agg/Std.

  Relaxed: return a_raw bit-identically (no state). Agg/Std: jerk-
  limited preview toward a Relaxed-like target, scaled to their
  desired gap. Launch, urgent, and non-slowing leads pass through.
  """

  def __init__(self):
    self.prev = None

  def reset(self, a=None):
    self.prev = None if a is None or not _finite(a) else float(a)

  def apply(self, a_raw, v_ego, lead=None, personality=None, fcw=False,
            dt=DT, desired_gap=None, a_prev=None, experimental_mode=False):
    # t304u: Relaxed is the reference. Identity, including type.
    if is_relaxed(personality, experimental_mode):
      return a_raw

    if not _finite(a_raw):
      self.prev = None
      return a_raw

    # Launch: never block a go. Zero is a coast, not a go — that is
    # exactly when we should start a slowdown response if the lead
    # has begun braking and the MPC has not yet.
    if a_raw > 0.0:
      self.prev = float(a_raw)
      return float(a_raw)

    present = _lead_present(lead)
    d_rel = getattr(lead, "dRel", float("nan")) if present else float("nan")
    v_lead = getattr(lead, "vLead", float("nan")) if present else float("nan")
    a_lead = getattr(lead, "aLeadK", getattr(lead, "aLead", float("nan"))) if present else float("nan")

    if _urgent(a_raw, v_ego, v_lead, d_rel, fcw, min_gap=FINAL_STOP_MIN_GAP):
      self.prev = float(a_raw)
      return float(a_raw)

    # A comfortable approach may have a raw MPC command at or below
    # -1.5 simply because Agg's native jerk factor is 0.5 — that is
    # the peak we want to lower, when kinematics still allow. Real
    # emergencies already passed through via _urgent (FCW, TTC<2,
    # gap under 3.5). If kinematics need <= -1.5 to hold desired,
    # _approach_target will ask for it.

    if (not present) or (not lead_is_slowing(v_lead, a_lead, v_ego)):
      self.prev = float(a_raw)
      return float(a_raw)

    dtc = dt if (_finite(dt) and dt > 0.0) else DT
    if self.prev is not None:
      prev = self.prev
    elif _finite(a_prev):
      prev = float(a_prev)
    else:
      prev = float(a_raw)

    a_target = _approach_target(v_ego, v_lead, a_lead, d_rel, desired_gap, a_raw)
    a_lead_f = float(a_lead) if _finite(a_lead) else 0.0

    below = _finite(d_rel, desired_gap) and float(d_rel) < float(desired_gap)
    v = float(v_ego) if _finite(v_ego) else 0.0
    vl = float(v_lead) if _finite(v_lead) else 0.0
    closing = (v - vl) > 0.05

    # Jerk: 0.30 for a mild slowdown. Rise toward 0.90 (Bae band top)
    # when the lead itself is braking hard — a 0.30 ramp takes 6.7 s
    # to match -2.0 and we fall behind, then slam.
    jerk = FINAL_STOP_JERK
    if a_lead_f < -0.50:
      jerk = min(0.90, max(FINAL_STOP_JERK, abs(a_lead_f) / 1.5))

    if below and closing:
      # Never weaken a raw that's already braking when we are short.
      shaped = min(float(a_raw), a_target)
    elif a_raw > a_target + 0.08:
      # Raw is late / milder: start earlier.
      shaped = a_target
    elif a_raw < a_target - 0.15 and not below:
      # Raw is harder than needed and we have room: lower the peak.
      shaped = a_target
    else:
      # MPC is already in the comfortable band — do not rate-limit a
      # good response down to 0.30 (that was falling behind -1.5/-2.0
      # leads and then slamming).
      shaped = float(a_raw)
      limited = shaped
      if limited > 0.0:
        limited = 0.0
      self.prev = float(limited)
      return float(limited)

    if shaped > 0.0:
      shaped = 0.0

    limited = _jerk_toward(prev, shaped, jerk, dtc)
    if limited > 0.0:
      limited = 0.0
    self.prev = float(limited)
    return float(limited)
