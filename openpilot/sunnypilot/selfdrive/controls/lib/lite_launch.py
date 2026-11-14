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
  - the driving model is not asking to stop (e2e shouldStop False), and
    the floor is capped at the model's e2e accel, so a model stop (red
    light, stop sign) still wins.
Not road-tested.
"""
from __future__ import annotations

ENABLED = True

V_LEAD_MIN = 0.25     # m/s: lead must really be rolling (not radar noise)
V_REL_MIN = 0.25      # m/s: lead pulling away from us
MOVED_MIN = 0.15      # m: lead has actually moved away from where it stopped
CONFIRM_N = 2         # planner cycles (0.1 s) the gate must hold before acting
D_REF_ALPHA = 0.2     # EMA of dRel while both are stopped
A_LEAD_MIN = 0.0      # m/s^2: lead not slowing
A_PLAN_MIN = -0.20    # m/s^2: stock plan not braking
LAUNCH_FLOOR = 0.40   # m/s^2: take-up floor while gated
LAUNCH_V_MAX = 1.5    # m/s (~3.4 mph): floor/quick start only right off the line
ARM_V = 0.05          # m/s: arm only from a standstill
GAP_TOL = 0.0         # m below the stop target still allowed (none)


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

  def gate(self, lead, v_ego: float, fcw: bool, a_target: float) -> bool:
    if not (lead is not None and lead.present):
      return False
    v_lead = float(lead.vLead)
    return (v_ego < LAUNCH_V_MAX and
            v_lead >= V_LEAD_MIN and
            v_lead - v_ego >= V_REL_MIN and
            float(lead.aLeadK) >= A_LEAD_MIN and
            float(lead.dRel) >= stop_target() - GAP_TOL and
            self.d_ref is not None and float(lead.dRel) >= self.d_ref + MOVED_MIN and
            not fcw and
            a_target > A_PLAN_MIN)

  def update(self, lead, v_ego: float, standstill: bool, fcw: bool, a_target: float,
             a_e2e: float | None = None, e2e_should_stop: bool = False) -> tuple[bool, float]:
    """Returns (quick_start, a_target). a_target is never lowered."""
    if not ENABLED:
      self.reset()
      return False, a_target
    lead_present = lead is not None and lead.present
    if (standstill or v_ego < ARM_V) and lead_present:
      self.armed = True
      if float(lead.vLead) < 0.1:
        d = float(lead.dRel)
        self.d_ref = d if self.d_ref is None else self.d_ref + D_REF_ALPHA * (d - self.d_ref)
    if v_ego >= LAUNCH_V_MAX or not lead_present:
      self.armed = False
      self.d_ref = None
    ok = bool(self.armed and not e2e_should_stop and self.gate(lead, v_ego, fcw, a_target))
    self.count = self.count + 1 if ok else 0
    self.active = self.count >= CONFIRM_N
    if not self.active:
      return False, a_target
    floor = LAUNCH_FLOOR
    if a_e2e is not None:
      floor = min(floor, float(a_e2e))  # the driving model's stop (e.g. red light) still wins
    return True, max(float(a_target), floor)
