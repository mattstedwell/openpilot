"""devc4-test3: comfortable launch take-up from a stopped lead.

Port of test2-dev LaunchStart (a9d39c14a9) plus the predictive
launch_go and 0.10 m/s² brake-release preview (ef17d6bef5). This
module is launch take-up only: it does NOT include test2's +8 ft
stopped gap, LeadDecel, 30%-PWR caps, cut-in, catch-up, or
final-stop/hold. Test3's C3X accel caps (Standard [1.6, 1.2, 0.8, 0.6],
Relaxed 80% of that, Aggressive that curve plus the +0.4 launch boost),
follow gaps, and stopped-gap extras (+5 / +5 / +10 ft) are unchanged
and are the cap this envelope must respect.

When a lead is present and ego is stopped or under ~10 mph:
  * Start moving as soon as the lead is actually departing. launch_go
    treats a clearly accelerating lead (a_lead >= 0.25, projected
    v_lead + a_lead*0.20 s above 0.10) as "go" on the first frame,
    including when v_lead is still under 0.10. A stopped lead (v≈0,
    a≈0) is always False, so there is no creep.
  * Two planner-frame confirm is kept for the 0.30 m/s² first step
    (radar-noise robustness). The first confirm frame is a 0.10 m/s²
    brake-release preview so longcontrol leaves Stopping and the plant
    is already coming off a hold when the step fires. Then a 0.4 m/s³
    ramp, never above the personality cap.
  * A single noisy go frame is only the preview and is abandoned if
    go drops. A false-start (lead moves then stops) hard-releases to
    the chosen plan. FCW, urgent braking, or a gap under 3.5 m also
    hard-release.
  * The path is inactive above ~10 mph (jerk-limited hand-off);
    launch_go is false for a new engagement above that window.

A 0.25 s time-to-roll (plant v > 0.05 m/s) is not comfortably
reachable: the host plant tau is 0.25 s and Toyota's actuator delay
is 0.05 s; with the 0.30 step and a plant already at rest, v > 0.05
takes ~0.36 s after the command. This design gets the request as
early as is safe and keeps the comfortable 0.30 / 0.4 take-up.

Nothing road-tested. panda and docs/SAFETY.md are untouched.
"""
from __future__ import annotations

import numpy as np

# Planner rate (DT_MDL).
DT = 0.05
MPH = 0.44704

# --- launch take-up (from test2-dev LaunchStart + ef17d6bef5) ---
LAUNCH_START_STEP = 0.30          # m/s²; first positive request after confirm
LAUNCH_JERK_UP = 0.40             # m/s³; lead-launch ramp
LAUNCH_GO_V_LEAD = 0.10           # m/s; lead must be clearly moving (~0.22 mph)
LAUNCH_GO_PULL_AWAY = 0.05        # m/s; v_lead - v_ego, gap growing
LAUNCH_GO_V_EGO_MAX = 4.5         # m/s (~10 mph)
LAUNCH_GO_MIN_GAP = 3.5           # m; safety floor (independent of stop extras)
LAUNCH_GO_V_CRUISE = 0.5          # m/s; no-lead set-speed margin
LAUNCH_GO_CONFIRM = 2             # planner frames of go before the 0.30 step
LAUNCH_GO_URGENT = -1.0           # m/s²; never lift an already-urgent plan
LAUNCH_GO_CATCH = 0.5             # m/s; hysteresis if ego just caught up
LAUNCH_GO_STOPPED = 0.05          # m/s; lead is treated as stopped (hard release)
LAUNCH_GO_A_LEAD = 0.25           # m/s²; clearly accelerating away
LAUNCH_GO_PREDICT = 0.20          # s; project v_lead + a_lead*this for early go
LAUNCH_PREVIEW = 0.10             # m/s²; first confirm frame (clears should_stop)
LAUNCH_SOFT_EXIT_FLOOR = -0.5     # m/s²; closing bleed floor (real stop uses hard)


def _finite(*xs) -> bool:
  return all(x is not None and np.isfinite(x) for x in xs)


def _lead_present(lead) -> bool:
  if lead is None:
    return False
  if bool(getattr(lead, "status", False)):
    return True
  if bool(getattr(lead, "present", False)):
    return True
  return False


def _jerk_toward(prev, target, jerk, dt) -> float:
  """Move prev toward target at |jerk|, no overshoot."""
  if target > prev:
    return min(target, prev + jerk * dt)
  if target < prev:
    return max(target, prev - jerk * dt)
  return float(target)


def _lead_departing(v_lead, v_ego, a_lead) -> bool:
  """True when a lead is clearly accelerating away, even if v_lead has
  not yet reached LAUNCH_GO_V_LEAD. Projects v_lead one short horizon
  ahead. A stopped lead with a≈0 is always False."""
  if not _finite(v_lead, v_ego, a_lead):
    return False
  if a_lead < LAUNCH_GO_A_LEAD:
    return False
  if v_lead < -1e-6:
    return False
  if v_lead < v_ego - 1e-6:
    return False
  v_pred = float(v_lead) + float(a_lead) * LAUNCH_GO_PREDICT
  return v_pred > LAUNCH_GO_V_LEAD


def launch_go(present, d_rel, v_lead, v_ego, v_cruise, fcw=False,
              already_active=False, a_lead=None) -> bool:
  """True when ego should start immediately: lead actually moving away
  (or clearly accelerating away), or no lead with set speed above ego.
  Fail closed. A stopped lead (v≈0 and not accelerating) is always
  False, so the standstill gap holds."""
  if fcw:
    return False
  if present:
    if not (_finite(d_rel, v_lead, v_ego)):
      return False
    if d_rel < LAUNCH_GO_MIN_GAP:
      return False
    departing = _lead_departing(v_lead, v_ego, a_lead)
    # A truly stopped lead never launches. A lead that is already
    # accelerating away may still have v_lead below LAUNCH_GO_STOPPED
    # on the first frame; departing is allowed through.
    if (not departing) and v_lead <= LAUNCH_GO_STOPPED:
      return False
    pulling = v_lead > LAUNCH_GO_V_LEAD and v_lead >= v_ego + LAUNCH_GO_PULL_AWAY
    if (pulling or departing) and v_ego <= LAUNCH_GO_V_EGO_MAX:
      return True
    if already_active and v_lead > LAUNCH_GO_V_LEAD and v_lead >= v_ego - LAUNCH_GO_CATCH:
      return True
    return False
  if not (_finite(v_ego, v_cruise)):
    return False
  if v_ego > LAUNCH_GO_V_EGO_MAX:
    return False
  return bool(v_cruise > v_ego + LAUNCH_GO_V_CRUISE)


def launch_hard(present, d_rel, v_lead, fcw=False, a_chosen=0.0, a_lead=None) -> bool:
  """Hard release: FCW, urgent chosen plan, stopped lead, critically close.

  A lead that is already clearly accelerating away is not treated as
  stopped, even if v_lead is still under LAUNCH_GO_STOPPED — otherwise
  the predictive go path would be cancelled on the same frame.
  """
  if fcw:
    return True
  if _finite(a_chosen) and a_chosen <= LAUNCH_GO_URGENT:
    return True
  if present:
    if _finite(v_lead) and v_lead <= LAUNCH_GO_STOPPED:
      # v_ego argument of _lead_departing is only a pull-away check;
      # passing v_lead itself means "not closing on us."
      if not _lead_departing(v_lead, v_lead, a_lead):
        return True
    if _finite(d_rel) and d_rel < LAUNCH_GO_MIN_GAP:
      return True
  return False


class LaunchStart:
  """One comfortable first step, then launch-jerk, all personalities.

  The first confirm frame is LAUNCH_PREVIEW (0.10 m/s²), which clears
  should_stop so longcontrol leaves Stopping and the plant starts
  coming off a brake hold. The second confirm frame is
  LAUNCH_START_STEP (0.30 m/s²). Subsequent frames ramp at
  LAUNCH_JERK_UP (0.4 m/s³) and never exceed the personality cap. A
  single noisy go frame is only the preview, not the 0.30 step; if go
  drops, we return the chosen plan and should_stop re-arms. A hard
  release returns the chosen plan immediately. Any other exit is
  jerk-limited.
  """

  def __init__(self):
    self.reset()

  def reset(self, a=None):
    self.prev = 0.0 if a is None or not _finite(a) else float(a)
    self.active = False
    self.go_count = 0

  @property
  def engaging(self) -> bool:
    """True during confirm preview and after the 0.30 step."""
    return bool(self.active or self.go_count > 0)

  def update(self, go, a_chosen, cap, jerk, dt,
             v_ego=float("nan"), v_lead=None, a_lead=None, hard=False) -> float:
    if not _finite(a_chosen):
      self.reset()
      return a_chosen
    j = jerk if (_finite(jerk) and jerk > 0.0) else LAUNCH_JERK_UP
    dtc = dt if (_finite(dt) and dt > 0.0) else DT

    def _cap(val):
      if _finite(cap) and cap > 0.0:
        return min(val, float(cap))
      return val

    if hard or a_chosen <= LAUNCH_GO_URGENT:
      self.active = False
      self.go_count = 0
      self.prev = float(a_chosen)
      return float(a_chosen)

    if not go and not self.active:
      self.go_count = 0
      self.prev = float(a_chosen)
      return float(a_chosen)

    if not go and self.active:
      target = float(a_chosen)
      closing = (_finite(v_ego, v_lead) and v_ego > v_lead - 0.10)
      if closing:
        match = 0.0
        if _finite(a_lead) and a_lead >= 0.0:
          match = float(a_lead)
        v_over = max(0.0, float(v_ego) - float(v_lead))
        target = min(target, match - 1.5 * v_over)
        target = max(target, LAUNCH_SOFT_EXIT_FLOOR)
      a = _cap(_jerk_toward(self.prev, target, j, dtc))
      if abs(a - float(a_chosen)) <= max(j * dtc, 1e-6):
        self.active = False
      self.prev = float(a)
      return float(a)

    # go is True
    self.go_count += 1
    if (not self.active) and self.go_count < LAUNCH_GO_CONFIRM:
      # Brake-release preview: lift a hold to LAUNCH_PREVIEW so
      # should_stop clears (a >= 0.1) one frame before the 0.30 step.
      # Do not step *down* if the chosen plan is already above it.
      preview = LAUNCH_PREVIEW
      if _finite(cap) and cap > 0.0:
        preview = min(preview, float(cap))
      a = preview if a_chosen < preview else float(a_chosen)
      a = _cap(a)
      self.prev = float(a)
      return float(a)

    step = LAUNCH_START_STEP
    if _finite(cap) and cap > 0.0:
      step = min(step, float(cap))

    if not self.active:
      self.active = True
      # Lift a hold or a tiny MPC crawl to the step; do not step *down*
      # if the chosen plan is already above it.
      a = step if a_chosen < step else float(a_chosen)
    else:
      if _finite(cap) and cap > 0.0:
        desired = float(cap) if a_chosen < cap else float(a_chosen)
      else:
        desired = max(float(a_chosen), step)
      closing = (_finite(v_ego, v_lead) and v_ego > v_lead - 0.10)
      if closing:
        match = 0.0
        if _finite(a_lead) and a_lead >= 0.0:
          match = float(a_lead)
        v_over = max(0.0, float(v_ego) - float(v_lead))
        desired = min(desired, match - 1.5 * v_over)
        desired = max(desired, LAUNCH_SOFT_EXIT_FLOOR)
      a = _jerk_toward(self.prev, desired, j, dtc)
    a = _cap(a)
    self.prev = float(a)
    return float(a)


class LeadLaunch:
  """Planner-side state: LaunchStart take-up only (no LeadDecel)."""

  def __init__(self):
    self.launch = LaunchStart()

  def reset(self, a=None):
    self.launch.reset(a)

  def apply(self, a_chosen, should_stop, v_ego, v_cruise, lead, fcw, dt,
            cap=float("inf"), jerk_up=LAUNCH_JERK_UP, v_ego_stopping=0.25):
    present = _lead_present(lead)
    d_rel = getattr(lead, "dRel", float("nan")) if present else float("nan")
    v_lead = getattr(lead, "vLead", float("nan")) if present else float("nan")
    a_lead = getattr(lead, "aLeadK", float("nan")) if present else float("nan")

    hard = launch_hard(present, d_rel, v_lead, fcw=fcw, a_chosen=a_chosen,
                       a_lead=a_lead if present else None)
    go = launch_go(present, d_rel, v_lead, v_ego, v_cruise, fcw=fcw,
                   already_active=self.launch.active,
                   a_lead=a_lead if present else None)
    a = self.launch.update(go, a_chosen, cap, jerk_up, dt,
                           v_ego=v_ego, v_lead=v_lead if present else None,
                           a_lead=a_lead if present else None, hard=hard)

    # should_stop follows get_accel_from_plan: v < vEgoStopping and a < 0.1
    if _finite(v_ego) and v_ego < v_ego_stopping:
      should_stop = bool(a < 0.1)
    return float(a), bool(should_stop)
