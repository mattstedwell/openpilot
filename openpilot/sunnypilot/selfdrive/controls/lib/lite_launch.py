"""dev-c4-sunnypilot-lite: quicker start after the lead pulls away (Matt: ~0.5 s).

The ONLY launch change on this branch. Stock sunnypilot waits until the
lead MPC plan reaches +0.1 m/s^2 (should_stop) before leaving Stopping, and
the plan rises slowly while the radar's aLeadK catches up. This module
lets the car start as soon as the lead is clearly pulling away:

  * quick_start: the planner's shouldStop is cleared, so longcontrol leaves
    Stopping for PID even while the plan is still below the stock +0.1
    should_stop threshold. longcontrol itself is unchanged.
  * a small take-up floor (LAUNCH_FLOOR) on the plan, only below
    LAUNCH_V_MAX, so the PCM gets a positive request right away.

Every gate is re-checked on every planner cycle and nothing is latched or
bled off: the moment one fails, the plan is the stock MPC/cruise/e2e min
again (this is the opposite of the Nov 13 LeadLaunch, which kept "go"
while the lead slowed and ignored a_lead < 0). Gates, all required:
  - ego armed from a standstill behind this lead and still below LAUNCH_V_MAX
  - lead present, moving (v_lead >= V_LEAD_MIN) and faster than ego by
    V_REL_MIN, not slowing (aLeadK >= A_LEAD_MIN), and actually MOVED_MIN
    farther away than where it sat while we were both stopped
  - all of the above for CONFIRM_N consecutive planner cycles (radar noise)
  - gap (dRel) at least the stock stop distance + the lite 10 ft extra
  - no FCW, and the final stock plan (min of lead-MPC, cruise and the
    driving model's e2e candidate) is not braking (a_target > A_PLAN_MIN)
  - (sunnypilot-lite-test-fix) the driving model does NOT gate the launch
    any more unless it plans a stop the lead does not explain (stop line,
    sign, red light; lite_fusion.model_stop_blocks), which disarms the
    launch until the next standstill. Before the fix the launch waited on
    the model's e2e shouldStop and was capped at its accel, so a slow or
    stuck model kept the car stopped behind a departing lead.
Not road-tested.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.controls.lib.lite_fusion import conservative_gap, lead_moving_agreed, model_stop_blocks

ENABLED = True

V_LEAD_MIN = 0.25     # m/s: lead must really be rolling (not radar noise)
V_REL_MIN = 0.25      # m/s: lead pulling away from us
MOVED_MIN = 0.15      # m: lead has actually moved away from where it stopped
CONFIRM_N = 2         # planner cycles (0.1 s) the gate must hold before acting
D_REF_ALPHA = 0.2     # EMA of dRel while both are stopped
A_LEAD_MIN = 0.0      # m/s^2: lead not slowing
A_PLAN_MIN = -0.20    # m/s^2: stock plan not braking
LAUNCH_FLOOR = 1.00   # m/s^2: take-up floor while gated (fix: was 0.40; Matt wants real accel within ~1 s)
LAUNCH_V_MAX = 1.5    # m/s (~3.4 mph): floor/quick start only right off the line
ARM_V = 0.05          # m/s: arm only from a standstill
GAP_TOL = 0.0         # m below the stop target still allowed (none)
# Pre-release (fix): at a standstill, as soon as the radar lead starts to roll
# (before the quick-launch gate is confirmed), swap the deep standstill hold
# (stopAccel -2.0) for a light brake hold PRE_HOLD. Still a braking request, the
# car does not move; it only shortens the brake release once the launch comes.
# If the lead does not go, shouldStop returns and the stock hold rebuilds.
PRE_ENABLED = True
PRE_V_LEAD = 0.15     # m/s
PRE_MOVED = 0.03      # m
PRE_HOLD = -0.5       # m/s^2 (a brake request)
LAUNCH_MIN_GAP = 4.0  # m: never quick-launch from a gap shorter than this
MISSING_MAX_S = 1.0   # s of radar dropout tolerated before disarming
UNBLOCK_GAP_M = 3.0   # m: a stopped lead this close re-allows a launch after a model block


def stop_target() -> float:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
  return float(long_mpc.STOP_DISTANCE + long_mpc.LEAD_STOP_EXTRA)


class LiteLaunch:
  def __init__(self):
    self.reset()

  def reset(self):
    self.armed = False
    self.active = False
    self.d_ref = None
    self.count = 0
    self.travel = 0.0          # ego travel (m) since the last standstill
    self.model_blocked = False  # model stops for something the lead does not explain (sticky per launch)
    self.pending = False        # pre-release: lead starting to roll, light hold
    self.missing = 0.0          # s the lead has been missing

  def lead_moved(self, lead) -> float:
    if self.d_ref is None or lead is None or not lead.present:
      return 0.0
    return float(lead.dRel) + self.travel - self.d_ref

  def gap_floor(self) -> float:
    """Gap the lead must have opened to: the stop target, or (fix) if we stopped
    short of it, where we actually stopped plus MOVED_MIN (never below
    LAUNCH_MIN_GAP). Before the fix a stop that ended short of the target (the
    model's e2e candidate can do that) blocked the quick launch for good."""
    if self.d_ref is None:
      return stop_target()
    return min(stop_target(), max(self.d_ref + MOVED_MIN, LAUNCH_MIN_GAP))

  def pre_gate(self, lead, fcw: bool, a_gate: float) -> bool:
    if not (lead is not None and lead.present) or fcw or self.d_ref is None:
      return False
    if not lead_moving_agreed(lead, None):  # radar-measured
      return False
    return (float(lead.vLead) >= PRE_V_LEAD and float(lead.aLeadK) >= A_LEAD_MIN and
            float(lead.dRel) >= self.d_ref + PRE_MOVED and float(lead.dRel) >= LAUNCH_MIN_GAP and
            a_gate > A_PLAN_MIN)

  def gate(self, lead, v_ego: float, fcw: bool, a_target: float, model_lead=None) -> bool:
    if not (lead is not None and lead.present):
      return False
    # radar-measured lead (vision no longer has to agree: lite_fusion.REQUIRE_MODEL_AGREE)
    if not lead_moving_agreed(lead, model_lead):
      return False
    v_lead = float(lead.vLead)
    d = conservative_gap(lead, model_lead)
    return (v_ego < LAUNCH_V_MAX and
            v_lead >= V_LEAD_MIN and
            v_lead - v_ego >= V_REL_MIN and
            float(lead.aLeadK) >= A_LEAD_MIN and
            d >= self.gap_floor() - GAP_TOL and
            self.d_ref is not None and float(lead.dRel) >= self.d_ref + MOVED_MIN and
            not fcw and
            a_target > A_PLAN_MIN)

  def update(self, lead, v_ego: float, standstill: bool, fcw: bool, a_target: float,
             a_e2e: float | None = None, e2e_should_stop: bool = False, model_lead=None,
             model_stop_x=None, a_plan_radar: float | None = None, dt: float = 0.05) -> tuple[bool, float]:
    """Returns (quick_start, a_target). a_target is never lowered.

    sunnypilot-lite-test-fix: the driving model's e2e shouldStop / accel no
    longer hold the launch back behind a radar lead that pulls away (they did,
    and the car waited for the model; see lite_fusion.model_stop_blocks). The
    model still wins when it plans a stop the lead does not explain (line /
    sign / light): then the launch is disarmed until the next standstill.
    a_plan_radar: the stock plan WITHOUT the model candidate (min of lead MPC
    and cruise), used for the "stock plan is not braking" gate.
    """
    if not ENABLED:
      self.reset()
      return False, a_target
    lead_present = lead is not None and lead.present
    if (standstill or v_ego < ARM_V) and lead_present:
      if not self.armed:
        self.model_blocked = False
      self.armed = True
      self.travel = 0.0
      if float(lead.vLead) < 0.1:
        if float(lead.dRel) < stop_target() + UNBLOCK_GAP_M:
          self.model_blocked = False  # a new departure from our queue lead
        d = float(lead.dRel)
        self.d_ref = d if self.d_ref is None else self.d_ref + D_REF_ALPHA * (d - self.d_ref)
    else:
      self.travel += max(float(v_ego), 0.0) * float(dt)
    # (fix) a brief radar dropout no longer throws away the standstill reference: only a
    # dropout longer than MISSING_MAX_S (or ego past LAUNCH_V_MAX) disarms
    self.missing = 0.0 if lead_present else self.missing + float(dt)
    if v_ego >= LAUNCH_V_MAX or self.missing > MISSING_MAX_S:
      self.armed = False
      self.d_ref = None
    if not lead_present:
      self.count = 0
      self.active = False
      self.pending = False
      return False, a_target
    if self.armed and model_stop_blocks(model_stop_x, float(lead.dRel), self.lead_moved(lead)):
      self.model_blocked = True
    a_gate = float(a_target) if a_plan_radar is None else float(a_plan_radar)
    ok = bool(self.armed and not self.model_blocked and self.gate(lead, v_ego, fcw, a_gate, model_lead))
    self.count = self.count + 1 if ok else 0
    self.active = self.count >= CONFIRM_N
    self.pending = bool(PRE_ENABLED and not self.active and self.armed and not self.model_blocked and
                        (standstill or v_ego < ARM_V) and self.pre_gate(lead, fcw, a_gate))
    if not self.active:
      return False, a_target
    return True, max(float(a_target), LAUNCH_FLOOR)
