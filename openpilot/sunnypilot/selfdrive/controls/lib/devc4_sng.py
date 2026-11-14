"""dev-c4-sng-test: automatic stop-and-go / city smoothing (Chill only).

Only MPC *inputs* change, in the direction that makes the plan smoother
and never less careful. Nothing is added on top of the plan afterwards:
the planner still takes min(lead MPC, cruise), clips it to the positive
cap, and braking is never limited. STOP_DISTANCE, the personality stop
extras and T_FOLLOW are unchanged, so the planned standstill gap behind
a stopped lead is exactly the one Matt has set.

Detection (StopAndGoDetector), no toggle:
  * enters when a lead is present, ego is under ~20 mph, and either we
    stopped (< 1 mph) behind a lead in the last 90 s, or both ego and
    the lead are crawling (< 10 mph);
  * exits when ego stays above ~30 mph for 3 s, the lead is gone for
    8 s, or 90 s pass with no stop and ego is above ~20 mph;
  * the output weight is rate-limited (0 -> 1 in 2 s) so weights blend
    in and out with no step.

While active (weight w), see SngConfig for the experiment knobs:
  * Relaxed-style braking: the MPC jerk / accel-change factor is at
    least Relaxed's (1.0), so Aggressive (0.5) brakes like Relaxed;
  * gentler accel: the positive accel ceiling is lowered to the Relaxed
    curve for every personality (a ceiling only; braking untouched);
  * smoother take-up: while we are accelerating behind a lead that is
    not slowing and the gap is above the desired gap, the jerk factor
    is raised a little more (FrogPilot uses separate accel / decel jerk
    the same way). It is dropped immediately when the lead slows, we
    close on it, or the gap is under desired, so braking is never
    softened by it.

  * keep the set gap: the MPC lead danger factor is raised 0.75 -> 1.0
    and the danger-zone slack cost 100 -> 1000, and the personality stop
    extra is planned from the lead's predicted stop. These only make the
    MPC brake earlier / keep more room; in host sims stock dev-c4 stopped
    2.4-3.5 m closer than the set gap when a lead braked to a stop, this
    stops within ~0.2 m of it.

City (lead present, full up to 45 mph, fades out 45 -> 55 mph): the
braking floor, danger factor/cost and predicted stop extra apply; the
accel ceiling is not lowered.

Launch: SafeLaunch MPC weighting is off by default (it added creep jerk
and cut creep gap in sims); the quicker start comes from the Toyota
brake-hold release in opendbc (launch_release.py), which needs Matt's
review before any push.

Nothing road-tested. panda safety and docs/SAFETY.md untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MPH = 0.44704

SNG_ENTER_V = 20.0 * MPH      # m/s; enter only under ~20 mph
SNG_EXIT_V = 30.0 * MPH       # m/s; leave above ~30 mph ...
SNG_EXIT_HOLD_S = 3.0         # ... held this long
SNG_STOP_V = 1.0 * MPH        # m/s; "stopped" for the stop memory
SNG_CRAWL_V = 10.0 * MPH      # m/s; ego and lead both crawling
SNG_STOP_MEMORY_S = 90.0      # s; a stop behind a lead keeps it armed
SNG_NO_LEAD_EXIT_S = 8.0      # s
SNG_BLEND_RATE = 0.5          # 1/s; weight 0 -> 1 in 2 s

CITY_V_FULL = 45.0 * MPH      # m/s
CITY_V_NONE = 55.0 * MPH      # m/s

RELAXED_JERK_FLOOR = 1.0      # stock Relaxed jerk factor
LEAD_DANGER_FACTOR_STOCK = 0.75  # long_mpc.LEAD_DANGER_FACTOR
DANGER_ZONE_COST_STOCK = 100.    # long_mpc.DANGER_ZONE_COST

# Urgency: the Relaxed-style braking floor (higher jerk cost = later,
# gentler onset) is faded out when braking is urgent, so a hard-braking
# lead always gets the personality's own (quicker) response.
URGENT_A_LEAD_BP = (1.5, 2.5)   # m/s² of lead decel (aLeadK)
URGENT_A_PLAN_BP = (1.5, 2.5)   # m/s² of decel in the previous MPC plan (first 2 s)

TAKEUP_A_LEAD_MIN = -0.3      # m/s²; lead slowing blocks the take-up smoothing
TAKEUP_CLOSING_MARGIN = 0.3   # m/s; v_lead must be >= v_ego - this
TAKEUP_GAP_MARGIN = 0.5       # m; gap must be above desired + this
TAKEUP_RATE_ON = 1.0          # 1/s; take-up smoothing blends in over 1 s
TAKEUP_RATE_OFF = 20.0        # 1/s; and drops in one 50 ms frame


@dataclass(frozen=True)
class SngConfig:
  enabled: bool = True
  jerk_floor: float = RELAXED_JERK_FLOOR   # Relaxed-style braking in SnG/city
  takeup_jerk_scale: float = 1.0           # >1: smoother take-up while accelerating
  relaxed_cap: bool = True                 # Relaxed accel ceiling in SnG
  city: bool = True                        # braking floor also in city traffic
  safe_launch: bool = False                # t309u gated launch weighting (planner); off: hold release only
  safe_launch_brake_free: bool = True      # t309u: also free braking for the rest of that launch
  stop_extra_predicted: bool = True        # plan the stop gap from the lead's predicted stop
  t_follow_min: float | None = None        # follow-time floor in SnG (None: personality's own)
  brake_onset_free: bool = False           # drop accel-change cost while the lead slows (SnG)
  urgency_gate: bool = True                # fade the braking floor out when braking is urgent
  danger_factor: float | None = 1.0        # raise MPC lead danger factor (0.75 stock) in SnG/city
  danger_cost_scale: float = 10.0          # raise the MPC danger-zone slack cost (100 stock) in SnG/city
  traffic_t_follow: float | None = None    # FrogPilot Traffic Mode reference only
  traffic_jerk_scale: float | None = None  # FrogPilot Traffic Mode reference only


# Chosen setup (see /workspace/handoff/dev-c4-sng-test.md).
DEFAULT_CONFIG = SngConfig()


def _f(x) -> float:
  try:
    return float(x)
  except (TypeError, ValueError):
    return float("nan")


def _lead_present(lead) -> bool:
  return lead is not None and bool(getattr(lead, "status", False) or getattr(lead, "present", False))


def _approach(x, target, rate_up, rate_down, dt):
  if target > x:
    return min(target, x + rate_up * dt)
  return max(target, x - rate_down * dt)


class StopAndGoDetector:
  def __init__(self):
    self.reset()

  def reset(self):
    self.active = False
    self.weight = 0.0
    self.t_since_stop = math.inf
    self.t_no_lead = math.inf
    self.t_fast = 0.0

  def update(self, v_ego, lead, dt) -> float:
    v = _f(v_ego)
    present = _lead_present(lead)
    v_lead = _f(getattr(lead, "vLead", float("nan"))) if present else float("nan")
    if not math.isfinite(v):
      self.reset()
      return 0.0
    self.t_no_lead = 0.0 if present else self.t_no_lead + dt
    if present and v < SNG_STOP_V:
      self.t_since_stop = 0.0
    else:
      self.t_since_stop += dt
    self.t_fast = self.t_fast + dt if v > SNG_EXIT_V else 0.0

    crawling = present and math.isfinite(v_lead) and v < SNG_CRAWL_V and v_lead < SNG_CRAWL_V
    if not self.active:
      if present and v < SNG_ENTER_V and (self.t_since_stop < SNG_STOP_MEMORY_S or crawling):
        self.active = True
    else:
      if (self.t_fast >= SNG_EXIT_HOLD_S or self.t_no_lead >= SNG_NO_LEAD_EXIT_S or
          (self.t_since_stop >= SNG_STOP_MEMORY_S and v > SNG_ENTER_V)):
        self.active = False
    self.weight = _approach(self.weight, 1.0 if self.active else 0.0, SNG_BLEND_RATE, SNG_BLEND_RATE, dt)
    return self.weight


def _ramp(x, bp):
  if not math.isfinite(x):
    return 1.0
  return min(max((x - bp[0]) / (bp[1] - bp[0]), 0.0), 1.0)


def urgency(lead, plan_a_min) -> float:
  """0 (calm) .. 1 (urgent) from the lead's decel and the previous plan.
  Fail closed (urgent) on bad values."""
  u = 0.0
  if _lead_present(lead):
    u = max(u, _ramp(-_f(getattr(lead, "aLeadK", float("nan"))), URGENT_A_LEAD_BP))
  if plan_a_min is not None:
    u = max(u, _ramp(-_f(plan_a_min), URGENT_A_PLAN_BP))
  return u


def city_weight(v_ego, lead) -> float:
  if not _lead_present(lead):
    return 0.0
  v = _f(v_ego)
  if not math.isfinite(v) or v >= CITY_V_NONE:
    return 0.0
  if v <= CITY_V_FULL:
    return 1.0
  return (CITY_V_NONE - v) / (CITY_V_NONE - CITY_V_FULL)


def takeup_ok(v_ego, lead, desired_gap, a_prev, fcw=False) -> bool:
  """Accelerating behind a lead that is pulling away and not slowing,
  with the gap above desired. Fail closed."""
  if fcw or not _lead_present(lead):
    return False
  d_rel = _f(getattr(lead, "dRel", float("nan")))
  v_lead = _f(getattr(lead, "vLead", float("nan")))
  a_lead = _f(getattr(lead, "aLeadK", float("nan")))
  v, g, a = _f(v_ego), _f(desired_gap), _f(a_prev)
  if not all(math.isfinite(x) for x in (d_rel, v_lead, a_lead, v, g, a)):
    return False
  return (a > 0.05 and a_lead >= TAKEUP_A_LEAD_MIN and v_lead >= v - TAKEUP_CLOSING_MARGIN and
          d_rel >= g + TAKEUP_GAP_MARGIN)


def lead_slowing_or_closing(v_ego, lead, desired_gap) -> bool:
  if not _lead_present(lead):
    return False
  d_rel = _f(getattr(lead, "dRel", float("nan")))
  v_lead = _f(getattr(lead, "vLead", float("nan")))
  a_lead = _f(getattr(lead, "aLeadK", float("nan")))
  v, g = _f(v_ego), _f(desired_gap)
  if not all(math.isfinite(x) for x in (d_rel, v_lead, a_lead, v)):
    return True
  return a_lead < TAKEUP_A_LEAD_MIN or v_lead < v - TAKEUP_CLOSING_MARGIN or (math.isfinite(g) and d_rel < g)


@dataclass
class SngOutput:
  weight: float = 0.0          # stop-and-go weight
  city: float = 0.0            # city weight
  jerk_factor: float | None = None
  cap: float = 0.0
  t_follow: float | None = None
  stop_extra_predicted: float = 0.0
  brake_free: bool = False
  urgency: float = 0.0
  danger_factor: float | None = None
  danger_cost: float | None = None


class StopAndGo:
  def __init__(self, config: SngConfig = DEFAULT_CONFIG):
    self.config = config
    self.detector = StopAndGoDetector()
    self.takeup = 0.0
    self.out = SngOutput()

  def reset(self):
    self.detector.reset()
    self.takeup = 0.0

  def update(self, v_ego, lead, base_jerk_factor, cap, relaxed_cap, desired_gap, a_prev, fcw, dt,
             base_t_follow=None, plan_a_min=None) -> SngOutput:
    c = self.config
    if not c.enabled:
      self.out = SngOutput(cap=cap)
      return self.out
    w = self.detector.update(v_ego, lead, dt)
    wc = city_weight(v_ego, lead) if c.city else 0.0
    wb = max(w, wc)
    u = urgency(lead, plan_a_min) if c.urgency_gate else 0.0
    if fcw:
      u = 1.0

    jf = float(base_jerk_factor)
    jf = jf + wb * (1.0 - u) * max(c.jerk_floor - jf, 0.0)
    ok = takeup_ok(v_ego, lead, desired_gap, a_prev, fcw)
    self.takeup = _approach(self.takeup, 1.0 if ok else 0.0, TAKEUP_RATE_ON, TAKEUP_RATE_OFF, dt)
    if not ok:
      self.takeup = 0.0  # never let take-up smoothing linger into braking
    jf *= 1.0 + w * self.takeup * (c.takeup_jerk_scale - 1.0)

    t_follow = None
    if c.t_follow_min is not None and base_t_follow is not None:
      t_follow = float(base_t_follow) + w * max(c.t_follow_min - float(base_t_follow), 0.0)
    brake_free = bool(c.brake_onset_free and w > 0.5 and lead_slowing_or_closing(v_ego, lead, desired_gap))
    if c.traffic_jerk_scale is not None:
      jf *= 1.0 + w * (c.traffic_jerk_scale - 1.0)
    if c.traffic_t_follow is not None and base_t_follow is not None:
      t_follow = float(base_t_follow) + w * (min(c.traffic_t_follow, base_t_follow) - float(base_t_follow))

    danger_factor = None
    if c.danger_factor is not None:
      danger_factor = LEAD_DANGER_FACTOR_STOCK + wb * max(c.danger_factor - LEAD_DANGER_FACTOR_STOCK, 0.0)

    out_cap = float(cap)
    if c.relaxed_cap:
      out_cap = out_cap - w * max(out_cap - float(relaxed_cap), 0.0)  # only ever lowers
    self.out = SngOutput(weight=w, city=wc, jerk_factor=jf, cap=min(out_cap, float(cap)), t_follow=t_follow,
                         stop_extra_predicted=wb if c.stop_extra_predicted else 0.0, brake_free=brake_free,
                         urgency=u, danger_factor=danger_factor,
                         danger_cost=DANGER_ZONE_COST_STOCK * (1.0 + wb * max(c.danger_cost_scale - 1.0, 0.0)))
    return self.out
