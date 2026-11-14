"""dev-c4-human: human-like city following (Phoenix arterials).

Goal (Matt, t351u): in stop-light-to-stop-light arterial traffic
(McDowell / Thomas / Camelback, 7th Ave - 44th St, 35-45 mph limits) the
car should follow like an experienced local commuter, for every
personality (Aggressive < Standard < Relaxed ordering kept).

Human targets and sources: /workspace/handoff/dev-c4-human.md.
  * Dense queues: HCM base saturation headway ~1.9 s front-to-front
    (1900 pc/h/ln) => ~16 m bumper gap at 25 mph; Zhu et al. 2016
    naturalistic medium-speed (41-65 km/h) mean headway 1.58 s; IDM
    city calibrations T 1.03-1.51 s (Kesting & Treiber 2008).
  * Standing queues: humans stop 1.4-3.7 m bumper to bumper. Matt's
    set stopping distance (6 m + personality extra) is LARGER and is
    kept: nothing here moves the standstill target closer.

What it changes (MPC inputs only; the plan is never raised above the
MPC lead plan, braking is never limited):
  1. City dense follow: while a lead is present, calm (not braking
     hard, not much slower than us, no FCW, plan not urgent) and we are
     between ~8 and 45 mph, T_FOLLOW is lowered toward a human dense-
     traffic curve (Standard = median commuter; Aggressive 0.10 s
     shorter, Relaxed 0.20 s longer, each never above its stock T).
     Faded in 4 -> 8 mph (launch untouched below 4 mph), out 45 -> 55.
     T returns to stock promptly (0.6 s/s) whenever the gate drops (lead
     brakes, closing, lost lead, FCW, urgency), so slowing for a
     braking/stopped lead starts at stock distance or earlier. T only
     shrinks slowly (0.08 s/s), so gaps close gradually like a person
     easing up, not a lunge.
  2. Relaxed-style braking floor (jerk factor >= 1.0) in city traffic,
     faded out when braking is urgent (lead decel / planned decel
     1.5 -> 2.5 m/s², FCW).
  3. Set-gap keeping: lead danger factor 0.75 -> 1.0 and danger-zone
     slack cost x10 (can only be raised) when the lead is slow (< 6 m/s)
     or braking, and the personality stop extra is planned from the
     lead's predicted stop (adds space only). From dev-c4-sng-test
     (d6045f6359), narrowed so similar-speed cut-ins are not braked hard.
  4. Lead hold-over (LeadHoldover): a close lead that briefly drops out
     of radarState is kept for up to 1 s at its last decel, so braking
     does not pause during a dropout. Only adds an obstacle; a held
     lead never raises FCW by itself.

The optional launch / accel part (HUMAN_LAUNCH, Toyota brake-hold
release + human take-up ceiling) lives in human_launch_cap() and is a
LAUNCH CHANGE that needs Matt's review before any push.

Nothing road-tested. panda safety and docs/SAFETY.md untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MPH = 0.44704

# --- city weight (lead present) --------------------------------------------
CITY_V_FULL = 45.0 * MPH
CITY_V_NONE = 55.0 * MPH

# --- dense follow -----------------------------------------------------------
# Human dense-traffic follow time T (s) for the MPC desired gap
# T*v + STOP_DISTANCE(6 m). Derived (see handoff notes):
#   25 mph: HCM 1.9 s saturation headway, 4.9 m vehicle -> 16.3 m gap -> T 0.92
#   35 mph: Zhu 2016 1.58 s headway -> 24.7 m gap -> T 1.20
#   45 mph: Zhu 1.45-1.58 s -> ~30 m gap -> T 1.20
DENSE_T_BP = (25.0 * MPH, 35.0 * MPH)
DENSE_T_MEDIAN = (0.95, 1.20)
DENSE_T_OFFSET = {"aggressive": -0.10, "standard": 0.0, "relaxed": 0.20}
DENSE_T_FLOOR = 0.85            # s; absolute floor (any personality)
DENSE_V_IN = (4.0 * MPH, 8.0 * MPH)   # fade in (no change below 4 mph)
DENSE_RATE_DOWN = 0.08          # s/s: T shrinks slowly (gradual close-up)
DENSE_RATE_UP = 0.6             # s/s: T returns to stock promptly (0.4 s offset in ~0.7 s)
# calm gate
CALM_A_LEAD_MIN = -0.75         # m/s²; lead braking harder -> stock T
CALM_CLOSING_MAX = 1.5          # m/s; v_ego - v_lead above this -> stock T
CALM_TTC_MIN = 6.0              # s; closing with TTC under this -> stock T

RELAXED_JERK_FLOOR = 1.0
LEAD_DANGER_FACTOR_STOCK = 0.75
DANGER_ZONE_COST_STOCK = 100.
URGENT_A_LEAD_BP = (1.5, 2.5)
URGENT_A_PLAN_BP = (1.5, 2.5)
# Set-gap keeping (danger factor/cost boost) only where the standstill gap
# is at stake: a slow lead or a braking lead. A similar-speed cut-in at
# 30 mph keeps stock danger weighting (no hard brake for a 10 m cut-in).
BOOST_V_LEAD_BP = (3.0, 6.0)    # m/s: full boost below 3, none above 6
BOOST_A_LEAD_BP = (0.3, 0.8)    # m/s² of lead decel: none below 0.3, full above 0.8
BOOST_RATE_UP = 20.0            # 1/s; on within one frame (slower fade-in braked later and harder in sims)
BOOST_RATE_DOWN = 0.5           # 1/s

# Lead hold-over: a lead that was close and not pulling away and then
# drops out of radarState for a moment is kept (predicted at its last
# decel, never accelerating) for up to HOLD_S, so braking continues
# through a brief radar/vision dropout. Only ever adds an obstacle.
HOLD_S = 1.0
HOLD_D_MAX = 40.0               # m
HOLD_PULLAWAY = 1.0             # m/s: lead faster than ego by more -> no hold
HOLD_DECEL_SCALE = 2.0          # held lead assumed to brake 2x its last (lagged) aLeadK ...
HOLD_DECEL_MAX = -4.0           # m/s² ... capped here

# --- optional launch / take-up (LAUNCH CHANGE, Matt reviews) ---------------
# Typical commuter accel from a stop: ~0.15 g mean, ~0.2 g peak (Wang et
# al. 2004, TRR 1883); 0.21 g mean over 3 s (Muttart 2010, SAE
# 2010-01-0062). Standard ~ median, ordering kept. Applied as a ceiling
# only while a lead is present and clearly pulling away with the gap at
# or above desired; the MPC lead plan still bounds it.
HUMAN_CAP_BP = (0.0, 15.0 * MPH, 30.0 * MPH)
HUMAN_CAP = {"aggressive": (1.75, 1.60, 0.0), "standard": (1.50, 1.40, 0.0), "relaxed": (1.30, 1.20, 0.0)}
TAKEUP_A_LEAD_MIN = -0.05
TAKEUP_V_LEAD_MIN = 0.15
TAKEUP_GAP_MARGIN = 0.05
# launch075 (t361u + steering: t_move ~0.75 s): while the same take-up gate
# holds and we are nearly stopped, let the planned accel rise faster than
# the comfort rise limit (0.6 m/s³), so the brake hold is released and
# drive torque arrives sooner after the lead moves. Faded out 1 -> 3 m/s.
# Only raises the *rise rate* of a positive request; the MPC lead plan
# still bounds the request and any decrease / braking passes through.
LAUNCH_JERK = 4.0               # m/s³ (comfort rise limit is 0.6)
LAUNCH_JERK_V_BP = (1.0, 3.0)   # m/s; full below 1, none above 3
# launch075 early go: under the same gate, while nearly stopped, leave the
# Stopping state once the MPC plans any real positive accel (> this)
# instead of the official 0.1 m/s² should_stop threshold.
LAUNCH_GO_A_MIN = 0.03          # m/s²
LAUNCH_GO_V_MAX = 0.3           # m/s (official should_stop speed)
# Repeated roll-stop gap gate: none of the launch aids (take-up ceiling,
# faster rise, lower jerk cost, early go) engage unless the gap is at
# least the personality's set stopping distance (minus a small radar
# tolerance). After a roll-stop that ended inside the set gap, the car
# waits for the lead to open the gap first and then launches stock.
SET_GAP_TOL = 0.05              # m
# Launch margin: from a gated quick launch until we pass LAUNCH_EP_V_END,
# the MPC plans this much extra space to the
# lead. A lead that creeps and stops right after we launch (the Nov 13
# near-miss shape) is then met at or beyond the set stopping distance
# even though we started sooner. Adds space only; after such a stop the
# standstill gap can be up to this much larger than the set distance.
LAUNCH_MARGIN = 1.2             # m
LAUNCH_EP_V_END = 5.0           # m/s
LAUNCH_MARGIN_V_BP = (0.0, 1.0)  # m/s; margin grows with ego speed, so it never delays the launch itself
# Stop anchor (Matt, real world: "rolling stop roll stop resulted in my car
# getting closer each time"). In one stop-and-go episode (cleared once we are
# clearly rolling, > LAUNCH_EP_V_END) the standstill gap behind a stopped lead
# is remembered after every stop (and the first one kept). At standstill /
# launching the MPC lead obstacle is moved by (last stop - TOL - set), so the
# launch sees the set gap, exactly as without it. While rolling toward a
# slowing / stopped lead it plans to stop no closer than the FIRST stop + AIM,
# so stops cannot ratchet closer. Adds lead space only; never more than
# STOP_ANCHOR_MAX over the set stopping distance.
STOP_ANCHOR_TOL = 0.05          # m
STOP_ANCHOR_MAX = 1.5           # m over the set stopping distance
STOP_ANCHOR_V = 0.05            # m/s, ego standstill
STOP_ANCHOR_V_LEAD = 0.3        # m/s, lead counts as stopped
STOP_ANCHOR_V_MOVED = 0.3       # m/s, ego rolled since the last stop (a new stop follows)
STOP_ANCHOR_A_LEAD = -0.2       # m/s², lead counts as slowing
STOP_ANCHOR_AIM = 0.15          # m, planned beyond the first stop (MPC stop overshoot is ~0.1-0.2 m)


@dataclass(frozen=True)
class HumanCityConfig:
  enabled: bool = True
  dense_follow: bool = True
  jerk_floor: float = RELAXED_JERK_FLOOR
  urgency_gate: bool = True
  danger_factor: float | None = 1.0
  danger_cost_scale: float = 10.0
  stop_extra_predicted: bool = True
  lead_holdover: bool = True
  boost_rate_up: float = BOOST_RATE_UP
  boost_on_decel: bool = True
  human_launch: bool = True      # LAUNCH CHANGE (planner ceiling); dev-c4-human-launch only, needs Matt review
  launch_jerk: float = LAUNCH_JERK  # LAUNCH CHANGE (launch075): gated faster take-up rise rate, m/s³
  launch_jerk_factor: float | None = None  # launch075: MPC jerk factor ceiling during the gated take-up
  lead_brake_jerk_factor: float | None = None  # launch075: MPC jerk factor ceiling near a slow/braking lead
  launch_go: bool = True         # launch075: early exit from Stopping under the gate
  set_gap_gate: bool = True      # launch075: roll-stop gap gate on every launch aid
  launch_margin: float = LAUNCH_MARGIN  # launch075: extra planned lead space after a quick launch, m
  stop_anchor: bool = True        # repeated roll-stop: later stops no closer than the first one


DEFAULT_CONFIG = HumanCityConfig()


def _f(x) -> float:
  try:
    return float(x)
  except (TypeError, ValueError):
    return float("nan")


def _interp(x, xp, fp):
  if x <= xp[0]:
    return fp[0]
  for i in range(1, len(xp)):
    if x <= xp[i]:
      t = (x - xp[i - 1]) / (xp[i] - xp[i - 1])
      return fp[i - 1] + t * (fp[i] - fp[i - 1])
  return fp[-1]


def _ramp(x, bp):
  if not math.isfinite(x):
    return 1.0
  return min(max((x - bp[0]) / (bp[1] - bp[0]), 0.0), 1.0)


def lead_present(lead) -> bool:
  return lead is not None and bool(getattr(lead, "status", False) or getattr(lead, "present", False))


def urgency(lead, plan_a_min) -> float:
  """0 calm .. 1 urgent; fails closed (urgent) on bad values."""
  u = 0.0
  if lead_present(lead):
    u = max(u, _ramp(-_f(getattr(lead, "aLeadK", float("nan"))), URGENT_A_LEAD_BP))
  if plan_a_min is not None:
    u = max(u, _ramp(-_f(plan_a_min), URGENT_A_PLAN_BP))
  return u


def city_weight(v_ego, lead) -> float:
  if not lead_present(lead):
    return 0.0
  v = _f(v_ego)
  if not math.isfinite(v) or v >= CITY_V_NONE:
    return 0.0
  if v <= CITY_V_FULL:
    return 1.0
  return (CITY_V_NONE - v) / (CITY_V_NONE - CITY_V_FULL)


def dense_t_target(v_ego, personality_name: str) -> float:
  t = _interp(_f(v_ego), DENSE_T_BP, DENSE_T_MEDIAN) + DENSE_T_OFFSET.get(personality_name, 0.0)
  return max(DENSE_T_FLOOR, t)


def calm_follow(v_ego, lead, fcw) -> bool:
  """Lead present and not braking / not much slower / no FCW. Fail closed."""
  if fcw or not lead_present(lead):
    return False
  d = _f(getattr(lead, "dRel", float("nan")))
  vl = _f(getattr(lead, "vLead", float("nan")))
  al = _f(getattr(lead, "aLeadK", float("nan")))
  v = _f(v_ego)
  if not all(math.isfinite(x) for x in (d, vl, al, v)) or d <= 0.0:
    return False
  closing = v - vl
  if al < CALM_A_LEAD_MIN or closing > CALM_CLOSING_MAX:
    return False
  if closing > 0.05 and d / closing < CALM_TTC_MIN:
    return False
  return True


def takeup_ok(v_ego, lead, desired_gap, fcw) -> bool:
  """Lead clearly pulling away, not slowing, gap >= desired. Fail closed.
  Same gates as SafeLaunch (81214b1956) launch_gate_ok."""
  if fcw or not lead_present(lead):
    return False
  d = _f(getattr(lead, "dRel", float("nan")))
  vl = _f(getattr(lead, "vLead", float("nan")))
  al = _f(getattr(lead, "aLeadK", float("nan")))
  v, g = _f(v_ego), _f(desired_gap)
  if not all(math.isfinite(x) for x in (d, vl, al, v, g)):
    return False
  return vl >= TAKEUP_V_LEAD_MIN and al >= TAKEUP_A_LEAD_MIN and vl > v and d >= g + TAKEUP_GAP_MARGIN


def human_launch_cap(v_ego, personality_name: str) -> float:
  vals = HUMAN_CAP.get(personality_name, HUMAN_CAP["standard"])
  return max(0.0, _interp(max(_f(v_ego), 0.0) if math.isfinite(_f(v_ego)) else 0.0, HUMAN_CAP_BP, vals))


def boost_weight(lead, on_decel=True) -> float:
  """0..1 set-gap boost: slow (or, with on_decel, braking) lead. Fails careful (1)."""
  if not lead_present(lead):
    return 0.0
  vl = _f(getattr(lead, "vLead", float("nan")))
  al = _f(getattr(lead, "aLeadK", float("nan")))
  if not (math.isfinite(vl) and math.isfinite(al)):
    return 1.0
  slow = 1.0 - _ramp(vl, BOOST_V_LEAD_BP)
  return max(slow, _ramp(-al, BOOST_A_LEAD_BP)) if on_decel else slow


class HeldLead:
  __slots__ = ("present", "status", "dRel", "vLead", "vLeadK", "aLeadK", "aLeadTau", "modelProb", "vRel", "held")

  def __init__(self, d, v, a, tau, v_ego):
    self.present = self.status = True
    self.dRel, self.vLead, self.vLeadK, self.aLeadK, self.aLeadTau = d, v, v, a, tau
    self.modelProb = 0.0   # a held lead never raises FCW by itself
    self.vRel = v - v_ego
    self.held = True


class LeadHoldover:
  def __init__(self):
    self.reset()

  def reset(self):
    self.last = None      # (d, v, a, tau) of the last real lead
    self.t_held = math.inf
    self.d = self.v = self.a = self.tau = float("nan")

  def update(self, lead, v_ego, dt):
    """Returns the real lead when present, a held lead during a short
    dropout, else the (absent) input."""
    v = _f(v_ego)
    if lead_present(lead):
      vals = tuple(_f(getattr(lead, k, float("nan"))) for k in ("dRel", "vLead", "aLeadK", "aLeadTau"))
      self.last = vals if all(math.isfinite(x) for x in vals) else None
      self.t_held = 0.0 if self.last is not None else math.inf
      if self.last is not None:
        self.d, self.v, self.a, self.tau = self.last[0], self.last[1], max(HOLD_DECEL_MAX, HOLD_DECEL_SCALE * min(self.last[2], 0.0)), self.last[3]
      return lead
    if self.last is None or not math.isfinite(v):
      return lead
    d0, v0 = self.last[0], self.last[1]
    if d0 > HOLD_D_MAX or v0 > v + HOLD_PULLAWAY or self.t_held + dt > HOLD_S:
      self.reset()
      return lead
    self.t_held += dt
    v_new = max(0.0, self.v + self.a * dt)
    self.d = max(0.1, self.d + 0.5 * (self.v + v_new) * dt - v * dt)
    self.v = v_new
    return HeldLead(self.d, self.v, self.a if self.v > 0.0 else 0.0, self.tau, v)


@dataclass
class HumanCityOutput:
  city: float = 0.0
  dense: float = 0.0
  urgency: float = 0.0
  jerk_factor: float | None = None
  t_follow: float | None = None
  danger_factor: float | None = None
  danger_cost: float | None = None
  stop_extra_predicted: float = 0.0
  cap: float = 0.0
  launch_jerk: float | None = None   # m/s³ rise-rate override for limit_accel_rise, None = stock
  launch_go: bool = False            # leave Stopping at a_mpc > LAUNCH_GO_A_MIN instead of 0.1
  launch_margin: float = 0.0         # m, extra planned lead space (adds space only)


class HumanCity:
  def __init__(self, config: HumanCityConfig = DEFAULT_CONFIG):
    self.config = config
    self.t_offset = 0.0   # s; <= 0, how far T is below stock
    self.boost = 0.0
    self.launch_ep = False
    self.launch_w = 0.0
    self.stop_anchor = None   # m, standstill gap of the last stop in this stop-and-go episode
    self.anchor_moved = False
    self.stop_anchor_first = None
    self.out = HumanCityOutput()

  def reset(self):
    self.t_offset = 0.0
    self.boost = 0.0
    self.launch_ep = False
    self.launch_w = 0.0
    self.stop_anchor = None
    self.anchor_moved = False
    self.stop_anchor_first = None

  def update(self, v_ego, lead, personality_name, base_jerk_factor, base_t_follow, cap, desired_gap,
             fcw, dt, plan_a_min=None, set_gap=None) -> HumanCityOutput:
    c = self.config
    if not c.enabled:
      self.reset()
      self.out = HumanCityOutput(cap=cap)
      return self.out
    v = _f(v_ego)
    wc = city_weight(v, lead)
    u = 1.0 if fcw else (urgency(lead, plan_a_min) if c.urgency_gate else 0.0)

    jf = float(base_jerk_factor)
    jf = jf + wc * (1.0 - u) * max(c.jerk_floor - jf, 0.0)

    # dense follow: target offset (<= 0) from stock T, rate limited
    t_follow = None
    tb = _f(base_t_follow)
    target = 0.0
    if c.dense_follow and math.isfinite(tb) and math.isfinite(v) and calm_follow(v, lead, fcw) and u <= 0.0:
      w_in = _ramp(v, DENSE_V_IN)
      target = wc * w_in * min(dense_t_target(v, personality_name) - tb, 0.0)
    if target < self.t_offset:
      self.t_offset = max(target, self.t_offset - DENSE_RATE_DOWN * dt)
    else:
      self.t_offset = min(target, self.t_offset + DENSE_RATE_UP * dt)
    if not lead_present(lead):
      self.t_offset = 0.0
    if math.isfinite(tb):
      t_follow = tb + min(self.t_offset, 0.0)

    out_cap = float(cap)
    launch_jerk = None
    launch_go = False
    gap_ok = True
    if c.set_gap_gate and set_gap is not None:
      d = _f(getattr(lead, "dRel", float("nan"))) if lead_present(lead) else float("nan")
      sg = _f(set_gap)
      gap_ok = math.isfinite(d) and math.isfinite(sg) and d >= sg - SET_GAP_TOL
    if c.human_launch and gap_ok and takeup_ok(v, lead, desired_gap, fcw) and u <= 0.0:
      out_cap = max(out_cap, human_launch_cap(v, personality_name))
      w_lj = 1.0 - _ramp(v, LAUNCH_JERK_V_BP)
      if c.launch_jerk > 0.0 and w_lj > 0.0:
        launch_jerk = w_lj * c.launch_jerk
      if c.launch_jerk_factor is not None and w_lj > 0.0:
        jf = jf - w_lj * max(jf - c.launch_jerk_factor, 0.0)
      launch_go = bool(c.launch_go and v < LAUNCH_GO_V_MAX)
      if v < LAUNCH_GO_V_MAX and c.launch_margin > 0.0:
        self.launch_ep = True
    # Ends only once we are clearly rolling. Not on FCW or a lost lead:
    # the margin only adds space, so keeping it is the careful side.
    if self.launch_ep and (not math.isfinite(v) or v > LAUNCH_EP_V_END):
      self.launch_ep = False
    # grows with speed, never shrinks inside an episode (a stop after a
    # quick launch keeps the margin, so it ends at or beyond the set gap)
    self.launch_w = max(self.launch_w, _ramp(v, LAUNCH_MARGIN_V_BP)) if self.launch_ep else 0.0
    launch_margin = c.launch_margin * self.launch_w

    # repeated roll-stop anchor (see STOP_ANCHOR_*)
    sg_a = _f(set_gap) if set_gap is not None else float("nan")
    if not c.stop_anchor or not math.isfinite(v) or v > LAUNCH_EP_V_END or not math.isfinite(sg_a):
      self.stop_anchor = None
      self.stop_anchor_first = None
    else:
      if v > STOP_ANCHOR_V_MOVED:
        self.anchor_moved = True
      if self.anchor_moved and v < STOP_ANCHOR_V and lead_present(lead):
        d_a = _f(getattr(lead, "dRel", float("nan")))
        vl_a = _f(getattr(lead, "vLead", float("nan")))
        if math.isfinite(d_a) and math.isfinite(vl_a) and vl_a < STOP_ANCHOR_V_LEAD and d_a >= sg_a - SET_GAP_TOL:
          self.stop_anchor = min(d_a, sg_a + STOP_ANCHOR_MAX)
          if self.stop_anchor_first is None:
            self.stop_anchor_first = self.stop_anchor
          self.anchor_moved = False
    if self.stop_anchor is not None and lead_present(lead):
      # At standstill / launching: the last stop gap, so the MPC sees us at the
      # set gap + TOL, the same as the plain launch case (no launch delay).
      target = self.stop_anchor - STOP_ANCHOR_TOL
      vl_a = _f(getattr(lead, "vLead", float("nan")))
      al_a = _f(getattr(lead, "aLeadK", 0.0))
      if self.anchor_moved and (not math.isfinite(vl_a) or vl_a < STOP_ANCHOR_V_LEAD or
                                (math.isfinite(al_a) and al_a < STOP_ANCHOR_A_LEAD)):
        # rolling toward a slowing / stopped lead: stop no closer than the
        # FIRST stop of this episode (+ AIM for the MPC's overshoot) -> no ratchet
        target = max(target, self.stop_anchor_first + STOP_ANCHOR_AIM)
      target = min(target, sg_a + STOP_ANCHOR_MAX)
      launch_margin = max(launch_margin, max(target - sg_a, 0.0))

    b_target = wc * boost_weight(lead, c.boost_on_decel)
    if b_target > self.boost:
      self.boost = min(b_target, self.boost + c.boost_rate_up * dt)
    else:
      self.boost = max(b_target, self.boost - BOOST_RATE_DOWN * dt)
    wb = self.boost
    if c.lead_brake_jerk_factor is not None:
      # Matt (t361u steering): firm, fast response to a slow / braking lead
      # beats smoothness, so the MPC jerk cost can only be lowered here.
      jf = jf - wb * max(jf - c.lead_brake_jerk_factor, 0.0)
    danger_factor = None
    if c.danger_factor is not None:
      danger_factor = LEAD_DANGER_FACTOR_STOCK + wb * max(c.danger_factor - LEAD_DANGER_FACTOR_STOCK, 0.0)
    self.out = HumanCityOutput(city=wc, dense=-self.t_offset, urgency=u, jerk_factor=jf, t_follow=t_follow,
                               danger_factor=danger_factor,
                               danger_cost=DANGER_ZONE_COST_STOCK * (1.0 + wb * max(c.danger_cost_scale - 1.0, 0.0)),
                               stop_extra_predicted=wc if c.stop_extra_predicted else 0.0, cap=out_cap,
                               launch_jerk=launch_jerk, launch_go=launch_go,
                               launch_margin=launch_margin)
    return self.out
