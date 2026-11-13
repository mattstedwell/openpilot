"""devc4-test3: chauffeur final stop and standstill hold.

The discomfort Matt reported is the jerk in the LAST FEW MPH as the
car comes to a full stop and into the standstill hold — not brake
release. On a 2021 Highlander Hybrid that is the regen-to-friction
blend handoff at low speed plus the official-dev final-stop lurch
(longcontrol builds toward stopAccel at 1.0 m/s²/s while still
rolling).

This module:
  * Holds commanded decel CONSTANT through the hybrid regen-friction
    blend band (~5-8 mph) so we are not also changing the request
    during that handoff.
  * In the last ~5 mph, a C2 quintic smootherstep tapers decel toward
    a near-zero chauffeur finish of 0.08 m/s² (in Matt's -0.05 to
    -0.1 window) — slightly negative so should_stop stays true and
    the car does not creep. Jerk is capped at 0.30 m/s³ (Bae et al.,
    Electronics 8(9):943, 2019, floor of the public-transport comfort
    band; well under the ~1 m/s³ seated JND of Müller et al., Proc.
    HFES 57(1):1211-1215, 2013).
  * In longcontrol, rolling in the stopping state follows that
    near-zero envelope (no grab toward stopAccel). Only after the
    car is fully at rest does hold pressure ramp to stopAccel at
    0.30 m/s²/s (Toyota official stoppingDecelRate). The final hold
    is still stopAccel, enough for a slope. No step at the
    pid → stopping or stopping → standstill switch.

Urgent stops pass through at full braking: FCW, a gap under the
min-gap floor (or 3.5 m if no floor is provided), or a TTC under
2 s. A hard mid-stop MPC request is still eased onto the last-5-mph
taper (the floor hook, applied after this, can always re-assert).
Positive requests (launch take-up) pass through bit-identically so
this never blocks a go. Not road-tested. panda and docs/SAFETY.md
are untouched.
"""
from __future__ import annotations

import math
import numpy as np

DT = 0.05
MPH = 0.44704

# --- last-few-mph chauffeur envelope ---
# Regen-friction blend band on Toyota hybrids (2021 Highlander Hybrid
# TSS2): hold the commanded decel constant between these speeds.
FINAL_STOP_BLEND_HI = 8.0 * MPH   # 3.576 m/s
FINAL_STOP_BLEND_LO = 5.0 * MPH   # 2.235 m/s
# Near-zero finish, in Matt's -0.05 to -0.1 window. Slightly negative
# so should_stop (a < 0.1) stays true and we do not creep.
FINAL_STOP_REST_DECEL = 0.08
# Default last-5-mph start (when no frozen blend-band value is held).
# 0.40 is a comfortable city-stop decel; the freeze uses the actual
# command at 8 mph instead when we have one.
FINAL_STOP_LAST5_START = 0.40
# 0.05 s Toyota TSS2 actuator delay: the rest value is the command
# we want at the wheels at v=0, so we finish it by REST_V.
FINAL_STOP_ACTUATOR_DELAY = 0.05
FINAL_STOP_REST_V = 0.15          # m/s; rest value reached by here

# --- jerk ---
# 0.30 = Bae 2019 comfort-band floor, well under Müller 2013 ~1 m/s³ JND.
FINAL_STOP_JERK = 0.30
# When a stopping lead is already closing and TTC is shrinking, jerk
# rises toward 0.90 (top of Bae's band, at the Müller JND) so we do
# not fall behind, then hands off to the urgent passthrough under 2 s.
# A comfortable approach stays at 0.30. In the last 5 mph of a
# comfortable stop we stay at 0.30 — that is the "very low jerk" Matt
# asked for.
FINAL_STOP_JERK_MAX = 0.90
FINAL_STOP_JERK_TTC_COMFORT = 6.0
FINAL_STOP_TTC = 2.0

# --- longcontrol hold ---
STOPPING_DECEL_RATE_STANDSTILL = 0.30  # m/s²/s; only after rest
STOPPING_DECEL_RATE_ROLLING = 0.30     # toward the envelope, while rolling
STOPPING_EASE_RATE = 0.30              # ease a stronger command toward it

# --- urgent passthrough ---
FINAL_STOP_URGENT_FULL = -1.5
FINAL_STOP_MIN_GAP = 3.5      # m; fallback if no personality floor given

# --- anticipation (stopping lead), used from the MPC ---
STOP_LEAD_ALEAD = -0.40
STOP_LEAD_TIME = 15.0
STOP_LEAD_TAU = 0.50
# Late reserved extra (meters) blended on v_ego, on top of test3's
# existing stop extras, to offset taper roll so the standstill floor
# is not eaten by the longer chauffeur finish. 0 above ~13 mph so
# moving follow is unchanged.
TAPER_COMP_M = 0.75
TAPER_COMP_V_FULL = 2.0
TAPER_COMP_V_NONE = 6.0


def _finite(*xs) -> bool:
  return all(x is not None and np.isfinite(x) for x in xs)


def _smootherstep(t: float) -> float:
  """C2 quintic smootherstep: 0 at t<=0, 1 at t>=1, s'=s''=0 at both ends."""
  if t <= 0.0:
    return 0.0
  if t >= 1.0:
    return 1.0
  return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _cosine_blend(x, x0, x1, y0, y1) -> float:
  if x <= x0:
    return y0
  if x >= x1:
    return y1
  t = (x - x0) / (x1 - x0)
  w = 0.5 * (1.0 - math.cos(math.pi * t))
  return y0 + (y1 - y0) * w


def _jerk_toward(prev, target, jerk, dt) -> float:
  if target > prev:
    return min(target, prev + jerk * dt)
  if target < prev:
    return max(target, prev - jerk * dt)
  return float(target)


def _lead_present(lead) -> bool:
  if lead is None:
    return False
  if bool(getattr(lead, "status", False)):
    return True
  if bool(getattr(lead, "present", False)):
    return True
  return False


def last5_envelope(v_ego: float, start_a: float = None) -> float:
  """Signed accel in the last 5 mph: C2 from start_a at 5 mph to
  -REST at 0, finishing the rest value by REST_V. Above 5 mph this
  returns start_a (the 5-8 mph band is held flat by the caller)."""
  start = -FINAL_STOP_LAST5_START if not _finite(start_a) else float(start_a)
  if start > 0.0:
    start = 0.0
  if not np.isfinite(v_ego):
    if v_ego > 0.0:
      return start
    if v_ego < 0.0:
      return -FINAL_STOP_REST_DECEL
    return float("nan")
  v = max(float(v_ego), 0.0)
  if v >= FINAL_STOP_BLEND_LO:
    return start
  if v <= FINAL_STOP_REST_V:
    if v <= 0.0:
      return -FINAL_STOP_REST_DECEL
    # C2 from -REST at 0 to the 5-mph-taper value at REST_V, so the
    # rest command is reached slightly early (actuator delay).
    t = v / FINAL_STOP_REST_V
    s = _smootherstep(t)
    rest = -FINAL_STOP_REST_DECEL
    at_rv = last5_envelope(FINAL_STOP_REST_V + 1e-6, start)
    return rest + (at_rv - rest) * s
  # C2 from value at REST_V (≈ rest, after the inner branch) — use
  # linear-in-smootherstep from start at 5 mph to rest at REST_V.
  t = (FINAL_STOP_BLEND_LO - v) / (FINAL_STOP_BLEND_LO - FINAL_STOP_REST_V)
  s = _smootherstep(t)
  return start + (-FINAL_STOP_REST_DECEL - start) * s


def final_stop_envelope(v_ego: float) -> float:
  """Default chauffeur envelope vs speed, for longcontrol and as a
  cap. Flat at -LAST5_START through the 5-8 mph blend band (no
  commanded change); C2 taper to -REST in the last 5 mph; -REST at
  a dead stop. Above 8 mph returns -LAST5_START as a mild cap only
  — the planner does not shape the approach above 8 mph."""
  if not np.isfinite(v_ego):
    if v_ego > 0.0:
      return -FINAL_STOP_LAST5_START
    if v_ego < 0.0:
      return -FINAL_STOP_REST_DECEL
    return float("nan")
  v = max(float(v_ego), 0.0)
  if v > FINAL_STOP_BLEND_HI:
    return -FINAL_STOP_LAST5_START
  if v >= FINAL_STOP_BLEND_LO:
    return -FINAL_STOP_LAST5_START  # flat 5-8 mph
  return last5_envelope(v, -FINAL_STOP_LAST5_START)


def stop_jerk(d_rel, v_ego, v_lead, a_raw=0.0) -> float:
  """0.30 when TTC is comfortable or we are not closing; cosine-blend
  up to 0.90 as TTC falls toward 2 s. Below 2 s the caller passes
  through. In the last 5 mph of a comfortable stop (not closing
  hard) this stays at 0.30."""
  if not (_finite(d_rel, v_ego, v_lead)):
    return FINAL_STOP_JERK
  v_rel = float(v_ego) - float(v_lead)
  if v_rel <= 0.05:
    return FINAL_STOP_JERK
  ttc = float(d_rel) / v_rel
  if ttc >= FINAL_STOP_JERK_TTC_COMFORT:
    return FINAL_STOP_JERK
  if ttc <= FINAL_STOP_TTC:
    return FINAL_STOP_JERK_MAX
  return _cosine_blend(ttc, FINAL_STOP_TTC, FINAL_STOP_JERK_TTC_COMFORT,
                       FINAL_STOP_JERK_MAX, FINAL_STOP_JERK)


def lead_is_stopping(v_lead, a_lead) -> bool:
  if not _finite(v_lead, a_lead):
    return False
  if v_lead <= 0.5 and a_lead <= 0.05:
    return True
  if a_lead > STOP_LEAD_ALEAD:
    return False
  if a_lead >= 0.0:
    return False
  return (float(v_lead) / max(-float(a_lead), 1e-6)) <= STOP_LEAD_TIME


def stopping_a_lead_tau(v_ego, a_lead, a_lead_tau, v_lead) -> float:
  """Persist a clearly-stopping lead's decel in the MPC prediction so
  the plan can start earlier and the last 5 mph can stay gentle. A
  highway slowdown that will not halt is left at the official tau."""
  if lead_is_stopping(v_lead, a_lead) and _finite(a_lead_tau):
    return float(min(a_lead_tau, STOP_LEAD_TAU))
  return a_lead_tau


def taper_comp_shift(v_ego) -> float:
  """Late reserved extra, meters, to offset chauffeur-taper roll.
  0 when v_ego is non-finite or above TAPER_COMP_V_NONE."""
  if not _finite(v_ego):
    return 0.0
  return float(np.interp(max(float(v_ego), 0.0),
                         [TAPER_COMP_V_FULL, TAPER_COMP_V_NONE],
                         [TAPER_COMP_M, 0.0]))


def _urgent(a_target, v_ego, v_lead, d_rel, fcw, min_gap=FINAL_STOP_MIN_GAP) -> bool:
  if fcw:
    return True
  if _finite(d_rel) and _finite(min_gap) and d_rel < min_gap:
    return True
  if _finite(d_rel, v_ego, v_lead):
    v_rel = float(v_ego) - float(v_lead)
    if v_rel > 0.05 and (float(d_rel) / v_rel) < FINAL_STOP_TTC:
      return True
  return False


def _is_stop_situation(v_ego, lead, a_target, should_stop) -> bool:
  if should_stop:
    return True
  if _finite(v_ego) and v_ego <= FINAL_STOP_BLEND_HI and _finite(a_target) and a_target < -0.05:
    return True
  present = _lead_present(lead)
  if not present:
    return False
  v_lead = getattr(lead, "vLead", float("nan"))
  a_lead = getattr(lead, "aLeadK", getattr(lead, "aLead", float("nan")))
  if not _finite(v_lead):
    return False
  if v_lead <= 0.5:
    return True
  return lead_is_stopping(v_lead, a_lead)


class FinalStopTaper:
  """Planner-side chauffeur taper and 5-8 mph command freeze.

  Above 8 mph: pass through (approach is unsmoothed here). Entering
  the 5-8 mph band freezes the current command and holds it constant
  until 5 mph. Below 5 mph: jerk-limited C2 taper to -REST. Urgent /
  FCW / close-gap / positive (launch) pass through bit-identically.
  """

  def __init__(self):
    self.prev = None
    self.blend_hold = None

  def reset(self, a=None):
    self.prev = None if a is None or not _finite(a) else float(a)
    self.blend_hold = None

  def apply(self, a_raw, v_ego, lead=None, fcw=False, dt=DT, should_stop=False,
            a_prev=None, min_gap=FINAL_STOP_MIN_GAP) -> float:
    if not _finite(a_raw):
      self.prev = None
      self.blend_hold = None
      return a_raw
    dtc = dt if (_finite(dt) and dt > 0.0) else DT
    present = _lead_present(lead)
    d_rel = getattr(lead, "dRel", float("nan")) if present else float("nan")
    v_lead = getattr(lead, "vLead", float("nan")) if present else float("nan")

    # Launch take-up and any non-braking request: never block a go,
    # and never clamp a positive command to 0.
    if a_raw >= 0.0:
      self.prev = float(a_raw)
      self.blend_hold = None
      return float(a_raw)

    if _urgent(a_raw, v_ego, v_lead, d_rel, fcw, min_gap=min_gap):
      self.prev = float(a_raw)
      self.blend_hold = None
      return float(a_raw)

    if not _is_stop_situation(v_ego, lead, a_raw, should_stop):
      self.prev = float(a_raw)
      self.blend_hold = None
      return float(a_raw)

    v = 0.0 if not _finite(v_ego) else max(float(v_ego), 0.0)

    if self.prev is not None:
      prev = self.prev
    elif _finite(a_prev):
      prev = float(a_prev)
    else:
      prev = float(a_raw)

    # 5-8 mph regen-friction band: freeze the command. Capture on
    # entry (the value we arrived with) and hold it constant.
    if FINAL_STOP_BLEND_LO < v <= FINAL_STOP_BLEND_HI:
      if self.blend_hold is None:
        hold = prev if _finite(prev) else float(a_raw)
        if hold > 0.0:
          hold = 0.0
        self.blend_hold = float(hold)
      limited = self.blend_hold
      self.prev = float(limited)
      return float(limited)

    if v <= FINAL_STOP_BLEND_LO:
      start = self.blend_hold if _finite(self.blend_hold) else (
        prev if _finite(prev) else -FINAL_STOP_LAST5_START)
      if start > 0.0:
        start = 0.0
      env = last5_envelope(v, start)
      # Ease a stronger (more negative) raw onto the chauffeur taper;
      # if the raw is already milder, do not fight it downward, but
      # never go positive.
      shaped = max(float(a_raw), env)
      if shaped > 0.0:
        shaped = 0.0
      j = stop_jerk(d_rel, v_ego, v_lead, a_raw)
      # In the last 5 mph of a comfortable stop, stay at the design
      # 0.30 even if TTC is shrinking a little — "very low jerk."
      # Urgent TTC already passed through above.
      if v <= FINAL_STOP_BLEND_LO:
        j = min(j, FINAL_STOP_JERK)
      limited = _jerk_toward(prev, shaped, j, dtc)
      if limited > 0.0:
        limited = 0.0
      self.prev = float(limited)
      return float(limited)

    # Above 8 mph on a stop situation: pass the raw approach through.
    # Remember it so we enter the blend band with a continuous value.
    self.blend_hold = float(a_raw) if a_raw <= 0.0 else None
    self.prev = float(a_raw)
    return float(a_raw)


def stopping_accel_step(output_accel, standstill, dt, v_ego=0.0,
                        stop_accel=-2.0) -> float:
  """One longcontrol stopping-state step, called only while
  output_accel > CP.stopAccel.

  Rolling: never weaken a strong command (<= -1.5). Otherwise move
  toward the chauffeur envelope (near-zero at v≈0) at 0.30 m/s²/s
  so there is no step at the pid → stopping switch and we do not
  grab toward stopAccel while still rolling. Standstill: only after
  the car is fully at rest, ramp from the current (near-zero) decel
  to stopAccel at 0.30 m/s²/s. The final hold is still stopAccel.
  """
  if not np.isfinite(output_accel):
    return output_accel
  output_accel = min(float(output_accel), 0.0)
  dtc = dt if (_finite(dt) and dt > 0.0) else 0.01
  if standstill:
    return output_accel - STOPPING_DECEL_RATE_STANDSTILL * dtc
  if output_accel <= FINAL_STOP_URGENT_FULL:
    return output_accel
  if not np.isfinite(v_ego):
    target = -FINAL_STOP_REST_DECEL
  else:
    target = final_stop_envelope(v_ego)
  if output_accel < target:
    return min(output_accel + STOPPING_EASE_RATE * dtc, target)
  if output_accel > target:
    return max(output_accel - STOPPING_DECEL_RATE_ROLLING * dtc, target)
  return output_accel
