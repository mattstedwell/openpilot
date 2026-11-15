"""sunnypilot-lite: stronger pull-away behind a lead that leaves a stop.

Matt (t372u/t374u): "when the car in front of me pulls away from the stop, my
car is not launching very quickly" ... "The first 10mph are more important".

Bottleneck on this tree (host sim, see handoff dev-c4-sunnypilot-lite.md):
there is no personality accel ceiling below ACCEL_MAX (2.0) here, so a ceiling
was never the limit. The limit is the lead MPC's own ramp: its accel-change
and jerk costs make the lead plan rise at only ~0.7-1.5 m/s^3 while the gap
is already opening, so the car sits well under the lead's accel for the first
2-3 s off the line.

This adds a launch take-up FLOOR on the final plan (it can only raise a
positive plan; it never lowers anything and never touches braking):
  target = FULL[personality] * w(v_ego) * dv_fade * gap_fade
  * FULL 2.0 / 1.8 / 1.6 m/s^2 (aggressive / standard / relaxed), all <= the
    openpilot / Toyota / panda 2.0 limit.
  * w = 1 from 0 to FULL_V (10 mph), C2 smootherstep to 0 at TAPER_END_V
    (22 mph); zero slope at both ends.
  * dv_fade: fades to 0 as ego speed comes within DV_FADE of the lead's.
  * proj_fade: 0 when a worst-case hard stop of the lead (-4.5) with a late,
    soft reaction from us (1.0 s, -2.0) would end inside the set gap + 0.5 m.
  * gap_fade: 0 at the MPC's own desired distance to this lead, 1 when the
    gap is GAP_FADE_M beyond it, so the boost only spends gap the MPC would
    also give up and never pushes inside the follow distance.
  * the floor rises from the previous output at most JERK_UP m/s^3, and it
    is capped at the driving model's e2e accel (a model stop still wins).
Gates (re-checked every cycle):
  * armed only from a standstill behind a present lead, ego < TAPER_END_V
  * radar-measured lead; if the model sees a lead it must agree it moves
    (lite_fusion); gaps use the smaller of radar and model distance
  * lead present, moving (>= V_LEAD_MIN), faster than ego, not slowing
    (aLeadK >= A_LEAD_MIN), gap >= the stop target (6.0 + 10 ft)
  * no FCW, model not asking to stop, stock plan not braking (> A_PLAN_MIN)
A slowing lead, FCW, a model stop, a braking stock plan or a lost lead
DISARMS the boost for the rest of that launch (it re-arms only at the next
standstill), so a lead that brakes after pulling away gets the stock plan.
Not road-tested.
"""
from __future__ import annotations

import math

from openpilot.cereal import log
from openpilot.sunnypilot.selfdrive.controls.lib.lite_fusion import conservative_gap, lead_moving_agreed, model_stop_blocks

# sunnypilot-lite-test-fix: ON again (Matt: launch too slow / leaves far too much
# room in city traffic). Radar-only gate; the driving model only disarms it when
# it plans a stop the lead does not explain (lite_fusion.model_stop_blocks).
ENABLED = True
MPH = 0.44704
_P = log.LongitudinalPersonality
FULL = {_P.aggressive: 2.0, _P.standard: 1.8, _P.relaxed: 1.6}
FULL_V = 10.0 * MPH        # m/s: full boost up to here
TAPER_END_V = 22.0 * MPH   # m/s: no boost from here
DV_FADE = 2.0              # m/s: fade as ego nears the lead's speed
GAP_FADE_M = 0.5           # m beyond the MPC desired distance for full boost
JERK_UP = 2.5              # m/s^3: floor rise limit from the previous output
V_LEAD_MIN = 1.5           # m/s (~3.4 mph): a real pull-away, not a creep or short roll
A_LEAD_MIN = -0.05         # m/s^2: lead not slowing
A_PLAN_MIN = -0.20         # m/s^2: stock plan not braking
ARM_V = 0.05               # m/s
SET_GAP_TOL = 0.05         # m
MISSING_MAX_S = 1.0   # s of radar dropout tolerated before disarming
MIN_REF_GAP = 4.0          # m: (fix) reference standstill gap never below this
# Worst-case stop check: if the lead slammed on the brakes right now
# (PROJ_LEAD_BRAKE) and we only started braking PROJ_REACT later at
# PROJ_EGO_BRAKE, we must still stop at least the full set gap (+ margin)
# behind it. The boost fades out over PROJ_FADE_M before that is violated, so
# it never spends the room a hard stop would need.
PROJ_LEAD_BRAKE = 4.5      # m/s^2
PROJ_EGO_BRAKE = 2.0       # m/s^2
PROJ_REACT = 1.0           # s
PROJ_MARGIN = 0.5          # m
PROJ_FADE_M = 1.0          # m


def _smootherstep01(x: float) -> float:
  x = min(max(x, 0.0), 1.0)
  return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def speed_weight(v_ego: float) -> float:
  v = float(v_ego)
  if not math.isfinite(v):
    return 0.0
  return 1.0 - _smootherstep01((v - FULL_V) / (TAPER_END_V - FULL_V))


def desired_gap(v_ego: float, v_lead: float, personality) -> float:
  """The lead MPC's comfort distance to this lead (its x_obstacle form)."""
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc as LM
  v_ego = max(float(v_ego), 0.0); v_lead = max(float(v_lead), 0.0)
  return float(LM.get_safe_obstacle_distance(v_ego, LM.get_T_FOLLOW(personality))
               - LM.get_stopped_equivalence_factor(v_lead) + LM.get_lead_stop_extra(v_lead))


def stop_target() -> float:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc as LM
  return float(LM.STOP_DISTANCE + LM.LEAD_STOP_EXTRA)


def projected_stop_gap(d: float, v_ego: float, v_lead: float) -> float:
  v_ego = max(float(v_ego), 0.0); v_lead = max(float(v_lead), 0.0)
  return float(d) + v_lead * v_lead / (2.0 * PROJ_LEAD_BRAKE) - (v_ego * PROJ_REACT + v_ego * v_ego / (2.0 * PROJ_EGO_BRAKE))


def boost_target(lead, v_ego: float, personality, d: float | None = None, ref_gap: float | None = None) -> float:
  """ref_gap: the standstill gap the hard-stop check must keep (default the
  stop target; the fix passes min(stop target, where we actually stopped))."""
  v_lead = float(lead.vLead)
  d = float(lead.dRel) if d is None else float(d)
  full = FULL.get(personality, FULL[_P.standard])
  dv = min(max((v_lead - v_ego) / DV_FADE, 0.0), 1.0)
  g = min(max((d - desired_gap(v_ego, v_lead, personality)) / GAP_FADE_M, 0.0), 1.0)
  ref = stop_target() if ref_gap is None else min(float(ref_gap), stop_target())
  pg = projected_stop_gap(d, v_ego, v_lead) - (ref + PROJ_MARGIN)
  gp = min(max(pg / PROJ_FADE_M, 0.0), 1.0)
  return full * speed_weight(v_ego) * dv * g * gp


class LiteLaunchBoost:
  def __init__(self):
    self.reset()

  def reset(self):
    self.missing = 0.0
    self.armed = False
    self.active = False
    self.floor = 0.0
    self.d_ref = None
    self.travel = 0.0

  def update(self, lead, v_ego: float, standstill: bool, fcw: bool, personality, a_target: float,
             a_prev: float, dt: float, a_e2e: float | None = None, e2e_should_stop: bool = False,
             model_lead=None, model_stop_x=None, a_plan_radar: float | None = None) -> tuple[bool, float]:
    """Returns (active, a_target). a_target is never lowered.

    sunnypilot-lite-test-fix: the model's e2e shouldStop / accel no longer
    disarm or cap the boost (they held the launch back behind a departing
    radar lead). A model stop the lead does not explain still disarms it for
    the rest of the launch. a_plan_radar = stock plan without the model
    candidate (lead MPC / cruise), used for the "not braking" gate.
    """
    self.active = False
    self.floor = 0.0
    if not ENABLED:
      self.armed = False
      return False, a_target
    present = lead is not None and lead.present
    v_ego = float(v_ego)
    # a brief radar dropout keeps the standstill reference (as lite_launch); longer -> disarm
    self.missing = 0.0 if present else self.missing + float(dt)
    if v_ego >= TAPER_END_V or self.missing > MISSING_MAX_S:
      self.armed = False
      self.d_ref = None
    if not present or v_ego >= TAPER_END_V:
      return False, a_target
    if standstill or v_ego < ARM_V:
      self.armed = True
      self.travel = 0.0
      if float(lead.vLead) < 0.1:
        self.d_ref = float(lead.dRel)
    else:
      self.travel += max(v_ego, 0.0) * float(dt)
    if not self.armed:
      return False, a_target
    a_lead = float(lead.aLeadK)
    a_gate = float(a_target) if a_plan_radar is None else float(a_plan_radar)
    moved = 0.0 if self.d_ref is None else float(lead.dRel) + self.travel - self.d_ref
    if fcw or a_lead < A_LEAD_MIN or a_gate <= A_PLAN_MIN or model_stop_blocks(model_stop_x, float(lead.dRel), moved):
      self.armed = False  # never boost toward a slowing lead / unexplained model stop / braking plan this launch
      return False, a_target
    if not lead_moving_agreed(lead, model_lead):  # radar-measured lead
      return False, a_target
    v_lead = float(lead.vLead)
    d = conservative_gap(lead, model_lead)
    ref = stop_target() if self.d_ref is None else min(stop_target(), max(self.d_ref, MIN_REF_GAP))
    if v_lead < V_LEAD_MIN or v_lead <= v_ego or d < ref - SET_GAP_TOL:
      return False, a_target
    tgt = boost_target(lead, v_ego, personality, d, ref_gap=ref)
    floor = min(tgt, max(float(a_prev), float(a_target)) + JERK_UP * dt)
    if floor <= a_target:
      return False, a_target
    self.active = True
    self.floor = floor
    return True, floor
