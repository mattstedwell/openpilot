"""devc4-test3: quicker, gated launch behind a departing lead (t309u).

Nothing here adds acceleration on top of the plan. The planned accel is
still the lead-following MPC (min'd with cruise and clipped to the cap),
so the car is never pushed toward the personality max and the MPC's
lead / danger-zone constraints stay in charge.

What changes is one MPC weight. Official openpilot already drops the
MPC "change from the previous accel" cost (A_CHANGE_COST) while the car
is at a standstill. On this tree that cost comes back the instant the
wheels turn, and it is what makes the first second of a launch creep
up at ~0.5 m/s²: the plan is tied to the previous (near-zero) request.
Here the standstill weighting is kept for the first ~5 mph of a launch,
but only while every launch gate holds:

  * a lead is present, values finite, no FCW
  * ego is under ~5 mph and slower than the lead (never out-runs it)
  * the lead is moving (v_lead >= 0.15 m/s) and not slowing
    (a_lead >= -0.05 m/s²)
  * the gap is above the desired follow gap + 0.05 m

For the rest of that same launch (until ego is back at a standstill or
above ~10 mph), the weight is also dropped whenever the lead slows
(a_lead < -0.05), we are closing (v_lead <= v_ego), or the gap is under
the desired gap. That only lets the MPC's *braking* respond sooner; it
never delays it, so braking is never slower than official. Ordinary
stops that did not start with a gated launch are untouched (official
weighting), so normal stopping feel does not change.

Nothing road-tested. panda safety and docs/SAFETY.md untouched.
"""
from __future__ import annotations

import math

MPH = 0.44704
SAFE_LAUNCH_V_EGO_MAX = 5.0 * MPH   # m/s; launch weighting only under ~5 mph
SAFE_LAUNCH_V_LEAD_MIN = 0.15       # m/s; lead clearly moving (not ~0)
SAFE_LAUNCH_A_LEAD_MIN = -0.05      # m/s²; any real slowing blocks launch
SAFE_LAUNCH_GAP_MARGIN = 0.05       # m; gap must already exceed desired
SAFE_BRAKE_V_EGO_MAX = 4.5          # m/s (~10 mph); quicker-brake band
SAFE_BRAKE_V_EGO_MIN = 1e-3         # m/s; standstill is already official


def _f(x) -> float:
  try:
    x = float(x)
  except (TypeError, ValueError):
    return float("nan")
  return x


def _finite(*xs) -> bool:
  return all(math.isfinite(_f(x)) for x in xs)


def _lead_present(lead) -> bool:
  return lead is not None and bool(getattr(lead, "status", False) or getattr(lead, "present", False))


def _lead_values(lead):
  return (_f(getattr(lead, "dRel", float("nan"))), _f(getattr(lead, "vLead", float("nan"))),
          _f(getattr(lead, "aLeadK", float("nan"))))


def desired_follow_gap(v_ego, v_lead, personality, experimental_mode=False) -> float:
  """Equal-speed desired bumper gap (m): T_FOLLOW*v + STOP_DISTANCE +
  the stop extra blended on v_lead, as the MPC uses it. At a standstill
  behind a stopped lead this is the planned standstill gap."""
  import numpy as np
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
    STOP_DISTANCE, LEAD_STOP_V_FULL, LEAD_STOP_V_NONE, get_T_FOLLOW, get_lead_stop_extra)
  v = max(_f(v_ego), 0.0) if _finite(v_ego) else 0.0
  extra = get_lead_stop_extra(personality, experimental_mode)
  blended = float(np.interp(_f(v_lead), [LEAD_STOP_V_FULL, LEAD_STOP_V_NONE], [extra, 0.0])) \
    if _finite(v_lead) else extra
  return float(get_T_FOLLOW(personality, experimental_mode, v) * v + STOP_DISTANCE + blended)


def launch_gate_ok(v_ego, lead, desired_gap, fcw=False) -> bool:
  """All launch gates hold (see module doc). Fail closed."""
  if fcw or not _lead_present(lead):
    return False
  d_rel, v_lead, a_lead = _lead_values(lead)
  v_ego, desired_gap = _f(v_ego), _f(desired_gap)
  if not _finite(v_ego, d_rel, v_lead, a_lead, desired_gap):
    return False
  if v_ego > SAFE_LAUNCH_V_EGO_MAX:
    return False
  if v_lead < SAFE_LAUNCH_V_LEAD_MIN:
    return False
  if a_lead < SAFE_LAUNCH_A_LEAD_MIN:
    return False
  if v_lead <= v_ego:
    return False
  return d_rel >= desired_gap + SAFE_LAUNCH_GAP_MARGIN


def brake_response_free(v_ego, lead, desired_gap) -> bool:
  """Under ~10 mph and moving, with the lead slowing, us closing, or the
  gap under desired: drop the change cost so braking builds immediately."""
  if not _lead_present(lead):
    return False
  d_rel, v_lead, a_lead = _lead_values(lead)
  v_ego, desired_gap = _f(v_ego), _f(desired_gap)
  if not _finite(v_ego, d_rel, v_lead, a_lead):
    return False
  if not (SAFE_BRAKE_V_EGO_MIN < v_ego < SAFE_BRAKE_V_EGO_MAX):
    return False
  if a_lead < SAFE_LAUNCH_A_LEAD_MIN or v_lead <= v_ego:
    return True
  return _finite(desired_gap) and d_rel < desired_gap


class SafeLaunch:
  """Planner state: a launch episode starts when the launch gates first
  hold and ends at a standstill or above SAFE_BRAKE_V_EGO_MAX."""

  def __init__(self):
    self.episode = False

  def reset(self):
    self.episode = False

  def prev_accel_constraint(self, prev_accel_constraint, v_ego, lead, desired_gap, fcw=False) -> bool:
    """MPC prev_accel_constraint for this frame. Only ever turns the
    official value off (standstill-style weighting); never on."""
    v = _f(v_ego)
    gate = launch_gate_ok(v_ego, lead, desired_gap, fcw)
    if gate:
      self.episode = True
    elif not math.isfinite(v) or v >= SAFE_BRAKE_V_EGO_MAX or (v <= SAFE_BRAKE_V_EGO_MIN and not prev_accel_constraint):
      self.episode = False
    if not prev_accel_constraint:
      return False
    if gate:
      return False
    if self.episode and brake_response_free(v_ego, lead, desired_gap):
      return False
    return True
