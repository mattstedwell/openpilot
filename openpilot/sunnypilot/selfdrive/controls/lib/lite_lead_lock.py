"""sunnypilot-lite-test2: stop-and-go "lead lock" (Matt, Nov 14).

Trigger: only right after both cars stop: ego at a standstill behind a STOPPED
radar lead (radar-measured, no vision). The car then locks onto that radar
track. While ego is at or below FULL_V (5 mph) the lock HOLDS the stopped gap
(where we stopped, or the stop target if smaller) with the radar lead's speed
(K_VEL) and accel as feed-forward plus a gap correction (K_GAP); the accel
feed-forward is low-pass filtered (A_LEAD_TAU) so radar aLeadK noise does not
cause jerks (Matt's decision: gap holding, not pure accel mirroring); positive accel capped at A_POS_MAX (the
Aggressive launch max). From FULL_V to END_V (6.5 mph) it blends (C1
smoothstep) back to the normal planner; above END_V the lock is released until
the next standstill. Hard overrides that always apply:

  * min(lock, normal plan) when the gap is under the stopped-gap floor minus
    GAP_MARGIN, TTC under TTC_MIN, FCW, the lead brakes harder than A_HARD,
    rolls backward, or is (nearly) stopped while we still roll. "Normal plan" =
    the stock plan WITHOUT the model candidate (lead MPC / cruise).
  * unlock (normal planner) if the radar lead is lost, is no longer
    radar-measured, changes track ID, or jumps closer by CUTIN_DROP_M (cut-in).
  * never positive unless the lead is really pulling away (vLead > vEgo +
    V_EPS and the gap is at least the floor); from a standstill only after a
    radar start check or the quick-launch gate, so radar noise at a stop cannot
    start the car.
  * positive accel never makes us outrun the lead (speed-match cap, T_MATCH)
    and never exceeds the speed from which we could still stop at the stopped
    gap if the lead slammed on the brakes now (SAFE_LEAD_DECEL, SAFE_DECEL,
    SAFE_LAG): no inching across roll-stop cycles, no gap lost when a lead
    brakes hard right after launching.
  * brakes at least enough to stop STOP_MARGIN behind the stopped gap behind
    where a decelerating lead will stop.
  * clamped to [A_MIN, A_MAX].
Not road-tested.
"""
from __future__ import annotations

import math

ENABLED = True
FULL_V = 5.0 * 0.44704      # m/s: full lock up to here (Matt: "until ego reaches 5 mph")
END_V = 6.5 * 0.44704       # m/s: planner only from here (lock released until the next standstill)
ARM_V = 0.05                # m/s: lock on only from a standstill
LEAD_STOPPED_V = 0.3        # m/s: the lead must be stopped when we lock on
LOCK_D_MAX = 25.0           # m
K_GAP = 0.30                # 1/s^2: gap-holding correction toward the stopped gap
K_VEL = 0.60                # 1/s: speed-matching correction (the lead's speed as feed-forward)
A_LEAD_TAU = 0.25           # s: low-pass on the radar aLeadK feed-forward (noise -> no jerks)
START_V_LEAD = 0.25         # m/s: one-cycle radar start check (lead rolling ...)
START_MOVED = 0.05          # m: ... and the gap already opened
V_EPS = 0.10                # m/s: lead must be faster than us by this to mirror positive accel
GAP_MARGIN = 0.30           # m below the stopped gap where the plan takes over
TTC_MIN = 3.0               # s
A_HARD = -2.0               # m/s^2: lead braking harder than this -> min with the plan
V_BACK = -0.05              # m/s: lead rolling backward
T_MATCH = 1.0               # s: speed-match horizon for the positive cap once rolling (human-like: ego
                            #    matches the lead's speed ~1 s later; shorter overshot the lead's speed and
                            #    cost up to ~2 m of gap when the lead braked hard right after launching)
T_MATCH_LO = 0.4            # s: ... from a standstill (quick first move), blended up to T_MATCH by T_MATCH_V
T_MATCH_V = 1.0             # m/s
STOP_MARGIN = 0.4           # m: aim this far behind the stopped gap when the lead stops (absorbs the
                            #    brake take-up overshoot so roll-stop cycles never creep closer)
SAFE_LEAD_DECEL = 4.5       # m/s^2: lead hard stop assumed by the always-can-stop cap (worst case)
SAFE_MARGIN = 0.05          # m: margin of the always-can-stop speed cap (small: does not delay the launch)
LEAD_STOP_DECEL = 2.5       # m/s^2: assumed lead decel for its stopping point (at least)
SAFE_DECEL = 2.0            # m/s^2: braking we plan on for the always-can-stop speed cap
SAFE_LAG = 0.8              # s: brake take-up allowance for that cap (Toyota jerk limit + lag; 0.4 let
                            #    roll-stop cycles creep ~0.4 m closer in sim, 0.6-0.8 holds them flat)
ACT_LAG = 0.25              # s: actuator lag allowance
CUTIN_DROP_M = 1.0          # m: dRel jumping closer by this in one cycle = a different object
A_MIN, A_MAX = -3.5, 2.0
A_POS_MAX = 2.0             # m/s^2: Aggressive max launch accel on this branch (boost FULL aggressive / ACCEL_MAX)


def _w(v: float) -> float:
  x = min(max((float(v) - FULL_V) / (END_V - FULL_V), 0.0), 1.0)
  return 1.0 - x * x * (3.0 - 2.0 * x)


class LeadLock:
  def __init__(self):
    self.reset()

  def reset(self):
    self.locked = False
    self.track_id = None
    self.d_ref = None
    self.d_anchor = None
    self.prev_d = None
    self.a_f = None
    self.active = False
    self.weight = 0.0

  def _unlock(self):
    self.locked = False
    self.track_id = None
    self.prev_d = None
    self.active = False
    self.weight = 0.0

  def update(self, lead, v_ego: float, standstill: bool, fcw: bool, a_target: float, a_plan_radar: float,
             launch_active: bool = False, dt: float = 0.05,
             stop_target: float | None = None) -> tuple[bool, float]:
    """Returns (active, a_target)."""
    self.active = False
    self.weight = 0.0
    if not ENABLED:
      self._unlock()
      return False, a_target
    v_ego = float(v_ego)
    present = lead is not None and bool(lead.present) and bool(getattr(lead, "radar", False))
    if not present:
      self._unlock()
      return False, a_target
    try:
      d = float(lead.dRel); v_lead = float(lead.vLead); a_lead = float(lead.aLeadK)
      tid = int(getattr(lead, "radarTrackId", -1))
    except Exception:
      self._unlock()
      return False, a_target
    if not (math.isfinite(d) and math.isfinite(v_lead) and math.isfinite(a_lead)):
      self._unlock()
      return False, a_target

    stopped = standstill or v_ego < ARM_V
    if not self.locked:
      # only right after both cars have stopped
      if stopped and abs(v_lead) < LEAD_STOPPED_V and 0.0 < d < LOCK_D_MAX:
        self.locked = True
        self.track_id = tid
        self.d_ref = d
        self.d_anchor = d  # gap at lock-on: the floor for every roll-stop cycle of this lock (no inching)
        self.prev_d = d
        self.a_f = None
      else:
        return False, a_target
    # unlock: other track, cut-in jump, past the lock speed
    if (self.track_id is not None and tid != self.track_id) or \
       (self.prev_d is not None and d < self.prev_d - CUTIN_DROP_M) or v_ego >= END_V:
      self._unlock()
      return False, a_target
    self.prev_d = d
    if stopped and abs(v_lead) < LEAD_STOPPED_V:
      self.d_ref = d  # still both stopped: this is the gap to hold
    floor_gap = self.d_anchor if stop_target is None else min(self.d_anchor, float(stop_target))
    hold_gap = floor_gap

    # positive only if the lead really pulls away; from a standstill a one-cycle radar check
    # (moving, opened the gap, accelerating) or the quick-launch gate
    pulling_away = v_lead > v_ego + V_EPS and d >= floor_gap - GAP_MARGIN
    start_ok = launch_active or (v_lead >= START_V_LEAD and d >= self.d_ref + START_MOVED and a_lead > 0.0)
    if stopped and not (pulling_away and start_ok):
      self.a_f = None
      return False, a_target  # locked but passive at a standstill: the stock hold / pre-release keep the car still

    # hold the stopped gap: feed-forward the lead's speed and (low-pass filtered) accel, so radar
    # aLeadK noise does not turn into jerks; the safety checks below use the raw accel
    alpha = min(max(float(dt), 0.0) / (A_LEAD_TAU + float(dt)), 1.0) if A_LEAD_TAU > 0.0 else 1.0
    self.a_f = a_lead if self.a_f is None else self.a_f + alpha * (a_lead - self.a_f)
    a_m = self.a_f + K_GAP * (d - hold_gap) + K_VEL * (v_lead - v_ego)
    if a_m > 0.0 and not (pulling_away and (start_ok or not stopped)):
      a_m = 0.0
    a_m = min(max(a_m, A_MIN), A_POS_MAX)
    v_rel = v_lead - v_ego
    ttc = d / -v_rel if v_rel < -1e-3 else math.inf
    lead_stopped = v_lead < LEAD_STOPPED_V  # lead (nearly) stopped while we still roll: the plan stops us
    if d < floor_gap - GAP_MARGIN or ttc < TTC_MIN or fcw or a_lead < A_HARD or v_lead < V_BACK or lead_stopped:
      a_m = min(a_m, float(a_plan_radar))
    if launch_active and v_lead >= v_ego and a_lead >= 0.0:
      a_m = max(a_m, min(float(a_target), A_POS_MAX))  # never weaker than the quick launch / boost
    # never outrun the lead: positive accel only up to what closes our speed to the lead's in T_MATCH
    # (radar vLead reacts faster than aLeadK when the lead brakes after a short roll)
    t_match = T_MATCH_LO + (T_MATCH - T_MATCH_LO) * min(max(v_ego / T_MATCH_V, 0.0), 1.0)
    if a_m > 0.0:
      a_m = min(a_m, max((v_lead - v_ego) / t_match, 0.0))
    # always able to stop at the stopped gap if the lead brakes now: speed no higher than v_safe, where
    # room = v^2/(2*SAFE_DECEL) + v*SAFE_LAG, room = gap + lead's stop distance (at LEAD_STOP_DECEL) - floor
    if a_m > 0.0:
      lead_stop_now = max(v_lead, 0.0) ** 2 / (2.0 * SAFE_LEAD_DECEL)
      room_now = d + lead_stop_now - (floor_gap + SAFE_MARGIN)
      v_safe = 0.0
      if room_now > 0.0:
        b = SAFE_DECEL
        v_safe = b * (-SAFE_LAG + math.sqrt(SAFE_LAG * SAFE_LAG + 2.0 * room_now / b))
      a_m = min(a_m, max((v_safe - v_ego) / t_match, 0.0))
    # no inching: brake at least enough to stop at the stopped gap behind where the lead will stop
    # (lead's stopping distance from its measured decel, or here if it is stopped / not decelerating)
    if self.a_f < -0.1 or v_lead < v_ego:  # filtered: radar noise must not toggle this brake on/off
      # conservative: the lead may stop at LEAD_STOP_DECEL even if radar still reports a softer decel
      # (aLeadK lags), and our actuators react ~ACT_LAG late
      lead_stop = v_lead * v_lead / (2.0 * max(-a_lead, -self.a_f, LEAD_STOP_DECEL)) if v_lead > 0.0 else 0.0
      room = d + lead_stop - (floor_gap + STOP_MARGIN) - v_ego * ACT_LAG
      a_req = -(v_ego * v_ego) / (2.0 * max(room, 0.05))
      a_m = min(a_m, a_req)
    w = _w(v_ego)
    a_out = w * a_m + (1.0 - w) * float(a_target)
    a_out = min(max(a_out, A_MIN), A_MAX)
    self.active = w > 0.0
    self.weight = w
    return self.active, a_out
