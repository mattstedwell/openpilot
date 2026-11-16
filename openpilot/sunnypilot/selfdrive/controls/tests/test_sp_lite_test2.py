"""sunnypilot-lite-test2: radar-only quicker launch, stop-and-go lead lock (gap holding),
steering-angle felt-g cap in turns, stopped gap unchanged from 4ba9a6967a. Unit + fuzz tests. Host only; NOT road-tested."""
from __future__ import annotations

import math
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc as LM
from openpilot.sunnypilot.selfdrive.controls.lib import lite_gap_open as GO
from openpilot.sunnypilot.selfdrive.controls.lib import lite_lead_lock as LK
from openpilot.sunnypilot.selfdrive.controls.lib import lite_turn_cap as TC

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3]
TREE = PKG.parent
FT = 0.3048
MPH = 0.44704
DT = 0.05
STOP_TGT = 6.0 + 10 * FT  # stock 6 m + 10 ft


def rl(d=9.05, v=0.0, a=0.0, present=True, radar=True, tid=7):
  return SimpleNamespace(present=present, dRel=d, vLead=v, aLeadK=a, radar=radar, radarTrackId=tid)


class TestStoppedGapUnchanged(unittest.TestCase):
  """Matt (Nov 14): keep the stopped gap behind a lead exactly as in 4ba9a6967a."""

  def test_constants(self):
    self.assertAlmostEqual(LM.LEAD_STOP_EXTRA, 10 * FT)
    self.assertAlmostEqual(LM.STOP_TAPER_COMP, 0.15)

  def test_file_identical_to_4ba9(self):
    try:
      out = subprocess.run(["git", "-C", str(TREE), "diff", "--stat", "4ba9a6967a", "--",
                            "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py",
                            "openpilot/sunnypilot/selfdrive/controls/lib/lite_e2e_stop.py",
                            "openpilot/selfdrive/controls/lib/longcontrol.py",
                            "openpilot/selfdrive/controls/controlsd.py"], capture_output=True, text=True, timeout=60)
    except Exception as e:  # pragma: no cover
      self.skipTest(f"git unavailable: {e}")
    if out.returncode != 0:
      self.skipTest(out.stderr)
    self.assertEqual(out.stdout.strip(), "")  # stopped gap, light/sign stop, soft final stop, controlsd untouched


class TestLeadLock(unittest.TestCase):
  def setUp(self):
    self.lk = LK.LeadLock()

  def step(self, ld, v_ego=0.0, standstill=None, fcw=False, a_t=0.0, a_plan=0.0, launch=False):
    ss = (v_ego < 0.01) if standstill is None else standstill
    return self.lk.update(ld, v_ego, ss, fcw, a_t, a_plan, launch_active=launch, dt=DT,
                          stop_target=STOP_TGT)

  def lock(self, d=9.05):
    for _ in range(5):
      self.step(rl(d=d))
    self.assertTrue(self.lk.locked)

  def test_locks_only_with_both_stopped(self):
    self.step(rl(d=9.0, v=2.0), v_ego=0.0)            # lead moving: no lock
    self.assertFalse(self.lk.locked)
    self.step(rl(d=9.0), v_ego=1.0, standstill=False)  # ego moving: no lock (only right after both stop)
    self.assertFalse(self.lk.locked)
    self.step(rl(d=9.0, radar=False))                  # vision-only lead: never
    self.assertFalse(self.lk.locked)
    self.step(rl(d=9.0))
    self.assertTrue(self.lk.locked)

  def test_passive_at_standstill_until_lead_pulls_away(self):
    self.lock()
    on, a = self.step(rl(d=9.05, v=0.05, a=0.3), a_t=-0.2)
    self.assertFalse(on); self.assertEqual(a, -0.2)
    on, a = self.step(rl(d=9.3, v=0.6, a=1.0), a_t=-0.2)
    self.assertTrue(on); self.assertGreater(a, 0.0)

  def test_feed_forward_follows_lead(self):
    self.lock()
    # ego rolling 1 m/s, lead 2 m/s, 10 m: positive cap by speed match / v_safe, but follows the lead's sign
    on, a = self.step(rl(d=10.5, v=2.0, a=1.0), v_ego=1.0, a_t=0.3, a_plan=0.3)
    self.assertTrue(on); self.assertGreater(a, 0.3)
    on, a = self.step(rl(d=10.5, v=1.0, a=-1.0), v_ego=1.0, a_t=0.3, a_plan=0.3)
    self.assertTrue(on); self.assertLessEqual(a, 0.0)  # positive dropped at once (speed match)
    for _ in range(int(0.5 / DT)):  # filtered accel feed-forward follows the lead's braking within ~0.5 s
      on, a = self.step(rl(d=10.5, v=1.0, a=-1.0), v_ego=1.0, a_t=0.3, a_plan=0.3)
    self.assertTrue(on); self.assertLess(a, -0.3)

  def test_positive_capped(self):
    self.lock()
    for v in (0.3, 1.0, 2.0):
      _, a = self.step(rl(d=30.0, v=v + 4.0, a=3.0), v_ego=v, a_t=0.0, a_plan=2.0)
      self.assertLessEqual(a, LK.A_POS_MAX + 1e-9)

  def test_never_positive_when_lead_not_pulling_away(self):
    self.lock()
    for ld in (rl(d=9.2, v=1.0, a=0.5), rl(d=9.2, v=0.9, a=0.5)):
      _, a = self.step(ld, v_ego=1.0, a_t=0.5, a_plan=0.5)
      self.assertLessEqual(a, 0.5 + 1e-9)  # the lock itself adds nothing (and v_lead <= v_ego + eps)

  def test_unlock_track_change_cutin_dropout(self):
    for ld in (rl(d=9.05, tid=8), rl(d=7.5), rl(present=False), rl(radar=False)):
      with self.subTest(ld=ld):
        self.setUp(); self.lock()
        self.step(rl(d=9.2, v=0.5, a=0.5), v_ego=0.3)
        on, a = self.step(ld, v_ego=0.3, a_t=-0.4)
        self.assertFalse(on); self.assertEqual(a, -0.4)
        self.assertFalse(self.lk.locked)

  def test_released_above_end_speed(self):
    self.lock()
    on, a = self.step(rl(d=15.0, v=5.0, a=1.0), v_ego=LK.END_V + 0.01, a_t=0.4)
    self.assertFalse(on); self.assertEqual(a, 0.4)
    self.assertFalse(self.lk.locked)

  def test_blend_weight(self):
    self.assertEqual(LK._w(0.0), 1.0)
    self.assertEqual(LK._w(LK.FULL_V), 1.0)
    self.assertEqual(LK._w(LK.END_V), 0.0)
    w = [LK._w(v) for v in np.linspace(0, 4, 200)]
    self.assertTrue(all(b <= a + 1e-12 for a, b in zip(w, w[1:])))

  def test_safety_overrides_take_plan_min(self):
    self.lock()
    cases = {"fcw": dict(fcw=True), "lead brakes hard": dict(ld=rl(d=10.0, v=1.5, a=-3.0)),
             "lead rolls back": dict(ld=rl(d=10.0, v=-0.2, a=0.0)), "inside floor": dict(ld=rl(d=8.6, v=1.5, a=0.5))}
    for name, kw in cases.items():
      with self.subTest(name):
        self.setUp(); self.lock()
        _, a = self.step(kw.get("ld", rl(d=10.0, v=1.5, a=0.5)), v_ego=1.0, fcw=kw.get("fcw", False), a_t=0.2, a_plan=-1.5)
        self.assertLessEqual(a, -1.5 + 1e-9, name)

  def test_always_can_stop_cap(self):
    # 0.3 m beyond the floor at 2 m/s with the lead pulling away: no more positive accel
    self.lock()
    _, a = self.step(rl(d=STOP_TGT + 0.3, v=2.2, a=1.0), v_ego=2.0, a_t=0.0, a_plan=1.0)
    self.assertLessEqual(a, 0.0)

  def test_no_inching_brakes_for_lead_stop(self):
    self.lock()
    _, a = self.step(rl(d=STOP_TGT + 0.5, v=0.5, a=-1.0), v_ego=1.5, a_t=0.0, a_plan=0.0)
    self.assertLess(a, -1.0)

  def test_accel_feedforward_filtered(self):
    # radar aLeadK noise (+-1 m/s^2 alternating) must not pass straight through to the command
    self.lock()
    outs = []
    for i in range(40):
      _, a = self.step(rl(d=10.0 + 0.02 * i, v=1.5, a=0.5 + (1.0 if i % 2 else -1.0)), v_ego=1.2, a_t=0.0, a_plan=1.0)
      outs.append(a)
    steps = np.abs(np.diff(outs[10:]))
    self.assertLess(float(np.max(steps)), 0.6)  # raw input swings 2.0 per cycle

  def test_left_blinker_hold_removed(self):
    self.assertFalse((PKG / "sunnypilot/selfdrive/controls/lib/lite_left_turn_hold.py").exists())
    for f in ("selfdrive/controls/controlsd.py", "selfdrive/controls/lib/longitudinal_planner.py"):
      self.assertNotIn("left_turn_hold", (PKG / f).read_text())

  def test_fuzz(self):
    r = np.random.default_rng(11)
    for _ in range(20000):
      if r.random() < 0.02:
        self.lk.reset()
      vals = [r.normal(8, 6), r.normal(0, 3), float('nan'), float('inf')]
      ld = rl(d=float(r.choice(vals)) if r.random() < 0.05 else float(r.uniform(0, 30)),
              v=float(r.uniform(-1, 6)), a=float(r.uniform(-5, 3)), present=bool(r.random() < 0.95),
              radar=bool(r.random() < 0.95), tid=int(r.integers(0, 3)) if r.random() < 0.02 else 7)
      v = float(r.uniform(0, 4)); a_t = float(r.uniform(-3.5, 2)); a_p = float(r.uniform(-3.5, 2))
      fcw = bool(r.random() < 0.05)
      on, a = self.step(ld, v_ego=v, standstill=v < 0.01, fcw=fcw, a_t=a_t, a_plan=a_p, launch=bool(r.random() < 0.3))
      self.assertTrue(math.isfinite(a))
      self.assertGreaterEqual(a, LK.A_MIN - 1e-9)
      self.assertLessEqual(a, max(a_t, LK.A_MAX) + 1e-9)
      if not on:
        self.assertEqual(a, a_t)
      if fcw and on:
        self.assertLessEqual(a, max(a_p, a_t) + 1e-9)


class TestTurnCap(unittest.TestCase):
  def test_straight_unlimited(self):
    self.assertEqual(TC.accel_limit(5.0, 0.0, 0.0, 0.0), math.inf)
    self.assertEqual(TC.cap_accel(1.5, 5.0, 0.0, 0.0, 0.0), 1.5)

  def test_braking_never_touched(self):
    for a in (-3.5, -1.0, 0.0):
      self.assertEqual(TC.cap_accel(a, 6.0, 0.1, 200.0, 50.0), a)

  def test_intersection_turn_no_accel(self):
    # 90 deg right turn at 12 mph: ~ -350 deg wheel
    v = 12 * MPH
    k = TC.curvature_from_angle(-350.0)
    self.assertGreater(TC.lat_g(v, k), TC.LAT_G_NO_ACCEL)
    self.assertEqual(TC.cap_accel(1.5, v, k, -350.0, 0.0), 0.0)

  def test_winding_in_holds_speed(self):
    v = 4.0
    k = TC.curvature_from_angle(-40.0)
    self.assertEqual(TC.cap_accel(1.0, v, k, -40.0, -60.0), 0.0)  # still winding in
    self.assertGreater(TC.cap_accel(1.0, v, k, -40.0, 60.0), 0.0)  # unwinding: some accel allowed

  def test_combined_g_bound(self):
    r = np.random.default_rng(5)
    for _ in range(5000):
      v = float(r.uniform(0, TC.V_MAX)); ang = float(r.uniform(-500, 500)); rate = float(r.uniform(-200, 200))
      k = TC.curvature_from_angle(ang)
      a = TC.cap_accel(float(r.uniform(-3.5, 2.0)), v, k, ang, rate)
      lg = TC.lat_g(v, k)
      if a > 0.0 and (lg >= TC.LAT_G_ON or abs(ang) >= TC.WIND_ANGLE_DEG):
        self.assertLess(lg, TC.LAT_G_NO_ACCEL)
        self.assertLessEqual(math.hypot(lg, a / TC.G), TC.G_TOTAL_MAX + 1e-9)

  def test_bad_inputs(self):
    for args in ((float('nan'), 0.1), (5.0, float('nan')), (5.0, float('inf'))):
      self.assertTrue(math.isfinite(TC.cap_accel(1.0, *args, 0.0, 0.0)))

  def test_planner_applies_last(self):
    src = (PKG / "selfdrive/controls/lib/longitudinal_planner.py").read_text()
    i = src.index("output_a_target = turn_cap_accel(")
    self.assertGreater(i, src.index("self.lead_lock.update("))
    self.assertGreater(i, src.index("self.gap_open.update("))
    self.assertGreater(src.index("self.gap_open.update("), src.index("self.lead_lock.update("))
    self.assertGreater(src.index("self.output_a_target = np.clip(output_a_target"), i)


class TestGapOpen(unittest.TestCase):
  """Post-launch soft accel cap: ramp to ~2 mph under the lead from 10 to 14 mph."""

  def setUp(self):
    self.go = GO.LiteGapOpen()
    self.p = LM.log.LongitudinalPersonality.aggressive if hasattr(LM, "log") else None
    from openpilot.cereal import log
    self.p = log.LongitudinalPersonality.aggressive
    self.std = log.LongitudinalPersonality.standard
    self.rel = log.LongitudinalPersonality.relaxed

  def step(self, ld, v_ego=8.0, standstill=False, fcw=False, a_t=1.0, a_prev=1.0, dt=DT,
           lock=False, boost=False, personality=None):
    return self.go.update(ld, v_ego, standstill, fcw, personality or self.p, a_t, a_prev, dt,
                          lock_active=lock, boost_active=boost)

  def arm_and_roll(self, t=1.2, v0=0.0, v1=8.0, d=14.0, v_lead=9.0, a_lead=0.0, a_t=1.0):
    """Standstill eligibility, then roll past the 10-14 mph ramp so the cap may apply."""
    self.step(rl(d=9.05), v_ego=0.0, standstill=True, a_t=-0.5, a_prev=-0.5)
    n = max(int(round(t / DT)), 1)
    a = a_t
    on = False
    for i in range(n):
      frac = (i + 1) / n
      v = v0 + (v1 - v0) * frac
      on, a = self.step(rl(d=d, v=v_lead, a=a_lead), v_ego=v, a_t=a_t, a_prev=a)
    return on, a

  def test_v_open_is_about_2_mph(self):
    self.assertAlmostEqual(GO.V_OPEN, 2.0 * MPH, places=3)
    # the cap's own ease-off is limited to A_SOFT; A_FLOOR is only a hard bound
    self.assertAlmostEqual(GO.A_SOFT, -0.10, places=2)
    self.assertAlmostEqual(GO.A_FLOOR, -0.30, places=2)
    self.assertGreater(GO.A_SOFT, GO.A_FLOOR)

  def test_follow_times_unchanged(self):
    from openpilot.cereal import log
    self.assertAlmostEqual(LM.get_T_FOLLOW(log.LongitudinalPersonality.aggressive), 1.25)
    self.assertAlmostEqual(LM.get_T_FOLLOW(log.LongitudinalPersonality.standard), 1.45)
    self.assertAlmostEqual(LM.get_T_FOLLOW(log.LongitudinalPersonality.relaxed), 1.75)

  def test_desired_gap_is_t_follow_v_plus_stop(self):
    from openpilot.cereal import log
    v = 20 * MPH
    for pers, T in ((log.LongitudinalPersonality.aggressive, 1.25),
                    (log.LongitudinalPersonality.standard, 1.45),
                    (log.LongitudinalPersonality.relaxed, 1.75)):
      self.assertAlmostEqual(GO.desired_follow_gap(v, pers), T * v + LM.STOP_DISTANCE)

  def test_inactive_at_standstill_and_below_ramp_start(self):
    on, a = self.step(rl(d=9.05), v_ego=0.0, standstill=True, a_t=-0.5, a_prev=-0.5)
    self.assertFalse(on); self.assertEqual(a, -0.5)
    # first few mph, still below RAMP_START_V: take-up is unchanged
    v = GO.RAMP_START_V - 0.05
    on, a = self.step(rl(d=10.0, v=3.0, a=1.0), v_ego=v, a_t=1.5, a_prev=-0.5)
    self.assertFalse(on); self.assertEqual(a, 1.5)
    self.assertFalse(self.go.episode)  # episode must not arm before 10 mph
    for _ in range(int(1.0 / DT)):
      on, a = self.step(rl(d=10.0, v=3.0, a=1.0), v_ego=v, a_t=1.5, a_prev=1.5)
    self.assertFalse(on); self.assertEqual(a, 1.5)
    self.assertFalse(self.go.episode)
    # lead-lock handoff speed (6.5 mph) is also still a no-op
    v_lock = 6.5 * MPH
    self.assertLess(v_lock, GO.RAMP_START_V)
    on, a = self.step(rl(d=10.0, v=v_lock + 1.0, a=1.0), v_ego=v_lock, a_t=1.5, a_prev=1.5)
    self.assertFalse(on); self.assertEqual(a, 1.5)
    self.assertFalse(self.go.episode)

  def test_min_on_top_of_lock_or_boost(self):
    """Cap may apply while lock/boost are still flagged; it only reduces accel."""
    v = 20 * MPH
    # lock still on, short gap, ego matching lead: cap applies as min()
    on, a = self.arm_and_roll(t=1.5, v1=v, d=13.0, v_lead=v, a_lead=0.0, a_t=1.5)
    # re-step with lock/boost flags; cap must still be allowed
    on, a = self.step(rl(d=13.0, v=v, a=0.0), v_ego=v, a_t=1.5, a_prev=a if on else 1.5, lock=True)
    self.assertTrue(on)
    self.assertLess(a, 1.5)
    self.assertGreaterEqual(a, GO.A_FLOOR - 1e-9)
    on, a = self.step(rl(d=13.0, v=v, a=0.0), v_ego=v, a_t=1.5, a_prev=a, boost=True)
    self.assertTrue(on)
    self.assertLess(a, 1.5)
    # lead brakes: drop even if lock/boost flags are set
    on, a = self.step(rl(d=13.0, v=v, a=-0.8), v_ego=v, a_t=0.4, a_prev=0.4, lock=True, boost=True)
    self.assertFalse(on); self.assertEqual(a, 0.4)

  def test_ramp_reaches_full_offset_at_14_mph(self):
    self.assertAlmostEqual(GO.RAMP_START_V, 10.0 * MPH, places=3)
    self.assertAlmostEqual(GO.RAMP_END_V, 14.0 * MPH, places=3)
    self.assertAlmostEqual(GO.v_open_scale(0.0), 0.0, places=6)
    self.assertAlmostEqual(GO.v_open_scale(GO.RAMP_START_V), 0.0, places=6)
    self.assertAlmostEqual(GO.v_open_offset(GO.RAMP_START_V), 0.0, places=6)
    # just below start: still zero; at 14 and above: full 2 mph
    self.assertAlmostEqual(GO.v_open_scale(GO.RAMP_START_V - 0.05), 0.0, places=6)
    self.assertAlmostEqual(GO.v_open_scale(6.5 * MPH), 0.0, places=6)
    self.assertAlmostEqual(GO.v_open_scale(GO.RAMP_END_V), 1.0, places=6)
    self.assertAlmostEqual(GO.v_open_offset(GO.RAMP_END_V), GO.V_OPEN, places=6)
    self.assertAlmostEqual(GO.v_open_offset(GO.RAMP_END_V + 1.0), GO.V_OPEN, places=6)
    # midpoint of the speed window is between but not 0.5 (smootherstep is S-shaped)
    mid = 0.5 * (GO.RAMP_START_V + GO.RAMP_END_V)
    s_mid = GO.v_open_scale(mid)
    self.assertGreater(s_mid, 0.15)
    self.assertLess(s_mid, 0.85)
    # monotonic and C1-flat at the ends (small step vs mid step)
    speeds = [GO.RAMP_START_V + i * 0.05 for i in range(int((GO.RAMP_END_V - GO.RAMP_START_V) / 0.05) + 1)]
    scales = [GO.v_open_scale(v) for v in speeds]
    self.assertTrue(all(scales[i] <= scales[i + 1] + 1e-12 for i in range(len(scales) - 1)))
    self.assertLess(scales[1] - scales[0], scales[len(scales)//2] - scales[len(scales)//2 - 1])

  def test_partial_cap_during_ramp_and_full_at_handoff(self):
    """By 14 mph the cap uses the full 2 mph offset; mid-ramp uses a smaller one."""
    # mid-ramp: ego matching lead, short gap -> cap, but less than full V_OPEN would demand
    v_mid = 0.5 * (GO.RAMP_START_V + GO.RAMP_END_V)
    des_mid = GO.desired_follow_gap(v_mid, self.p)
    on, a = self.arm_and_roll(t=1.2, v1=v_mid, d=0.7 * des_mid, v_lead=v_mid, a_lead=0.0, a_t=1.5)
    self.assertTrue(on)
    self.assertLess(a, 1.5)
    self.assertGreaterEqual(a, GO.A_FLOOR - 1e-9)
    # already sitting the mid-ramp offset under the lead: no cap
    self.go.reset()
    mid_off = GO.v_open_offset(v_mid)
    on, a = self.arm_and_roll(t=1.2, v1=v_mid, d=0.7 * des_mid, v_lead=v_mid + mid_off + 0.2, a_lead=0.2, a_t=1.5)
    self.assertFalse(on); self.assertEqual(a, 1.5)
    # at 14 mph, sitting only a mid-ramp offset under the lead is no longer enough
    self.go.reset()
    v = GO.RAMP_END_V
    des = GO.desired_follow_gap(v, self.p)
    on, a = self.arm_and_roll(t=1.5, v1=v, d=0.7 * des, v_lead=v + mid_off, a_lead=0.0, a_t=1.5)
    self.assertTrue(on)
    self.assertLess(a, 1.5)
    # but sitting the full 2 mph under is enough
    self.go.reset()
    on, a = self.arm_and_roll(t=1.5, v1=v, d=0.7 * des, v_lead=v + GO.V_OPEN + 0.2, a_lead=0.2, a_t=1.5)
    self.assertFalse(on); self.assertEqual(a, 1.5)

  def test_offset_shrinks_as_gap_approaches_target(self):
    """Full ~2 mph when well short; 0 at 100% of T_FOLLOW*v + STOP; smooth in between."""
    v = GO.RAMP_END_V
    des = 20.0
    self.assertAlmostEqual(GO.SHORT_FRAC, 0.90, places=2)
    self.assertAlmostEqual(GO.gap_short_scale(0.0, des), 1.0, places=6)
    self.assertAlmostEqual(GO.gap_short_scale(GO.SHORT_FRAC * des, des), 1.0, places=6)
    self.assertAlmostEqual(GO.gap_short_scale(des, des), 0.0, places=6)
    self.assertAlmostEqual(GO.gap_short_scale(des + 1.0, des), 0.0, places=6)
    mid = 0.5 * (GO.SHORT_FRAC + 1.0) * des
    s_mid = GO.gap_short_scale(mid, des)
    self.assertGreater(s_mid, 0.15)
    self.assertLess(s_mid, 0.85)
    # monotonic
    gaps = [GO.SHORT_FRAC * des + i * 0.1 for i in range(int((des - GO.SHORT_FRAC * des) / 0.1) + 1)]
    scales = [GO.gap_short_scale(g, des) for g in gaps]
    self.assertTrue(all(scales[i] >= scales[i + 1] - 1e-12 for i in range(len(scales) - 1)))
    # combined with the speed ramp: at 14 mph, 100% of target -> 0 offset
    self.assertAlmostEqual(GO.v_open_offset(v, des, des), 0.0, places=6)
    self.assertAlmostEqual(GO.v_open_offset(v, 0.5 * des, des), GO.V_OPEN, places=6)
    # below ramp start, still 0 even if the gap is short
    self.assertAlmostEqual(GO.v_open_offset(GO.RAMP_START_V, 0.5 * des, des), 0.0, places=6)

  def test_full_offset_when_well_short_partial_near_target(self):
    v = 20 * MPH
    des = GO.desired_follow_gap(v, self.p)
    # well short (70%): cap applies
    on, a = self.arm_and_roll(t=1.5, v1=v, d=0.70 * des, v_lead=v, a_lead=0.0, a_t=1.5)
    self.assertTrue(on)
    self.assertLess(a, 1.5)
    self.assertGreaterEqual(a, GO.A_FLOOR - 1e-9)
    a_short = a
    # at 95% the offset is tapering, so the cap must not be harsher than the well-short case
    self.go.reset()
    on, a = self.arm_and_roll(t=1.5, v1=v, d=0.95 * des, v_lead=v, a_lead=0.0, a_t=1.5)
    if on:
      self.assertGreaterEqual(a, a_short - 1e-9)
      self.assertLess(a, 1.5)
    # exactly at the target: no cap
    self.go.reset()
    on, a = self.arm_and_roll(t=1.5, v1=v, d=des, v_lead=v, a_lead=0.0, a_t=1.5)
    self.assertFalse(on); self.assertEqual(a, 1.5)

  def test_turns_off_at_target_and_may_reengage_on_a_later_sag(self):
    """At 100% of target the cap releases; the episode stays armed for a later sag."""
    v = 20 * MPH
    des = GO.desired_follow_gap(v, self.p)
    on, a = self.arm_and_roll(t=1.5, v1=v, d=0.70 * des, v_lead=v, a_lead=0.0, a_t=1.5)
    self.assertTrue(on)
    self.assertTrue(self.go.held)
    self.assertTrue(self.go.episode)
    # reach the target: jerk-limited release, then inactive but still in the episode
    off = False
    for _ in range(int(3.0 / DT)):
      on, a = self.step(rl(d=des + 0.05, v=v, a=0.0), v_ego=v, a_t=1.5, a_prev=a)
      if not on:
        off = True
        self.assertEqual(a, 1.5)
        break
    self.assertTrue(off)
    self.assertTrue(self.go.episode)
    self.assertFalse(self.go.held)
    # a later sag, without a new standstill, may ease off again
    on, a = self.step(rl(d=0.70 * des, v=v, a=0.0), v_ego=v, a_t=1.5, a_prev=1.5)
    self.assertTrue(on)
    self.assertLess(a, 1.5)
    self.assertGreaterEqual(a, GO.A_FLOOR - 1e-9)

  def test_caps_when_gap_short_and_closing(self):
    # 20 mph, Aggressive target ~ T*v+6 ≈ 17.2 m; gap 13 m; ego matching lead speed
    v = 20 * MPH
    des = GO.desired_follow_gap(v, self.p)
    self.assertGreater(des, 16.0)
    on, a = self.arm_and_roll(t=1.5, v1=v, d=13.0, v_lead=v, a_lead=0.0, a_t=1.2)
    self.assertTrue(on)
    self.assertLess(a, 1.2)
    self.assertGreaterEqual(a, GO.A_FLOOR - 1e-9)

  def test_never_below_floor_and_never_raises(self):
    v = 20 * MPH
    on, a = self.arm_and_roll(t=2.0, v1=v, d=10.0, v_lead=v - 1.0, a_lead=0.0, a_t=1.5)
    # ego faster than lead, short gap: cap applies, not below the soft ease-off
    self.assertTrue(on)
    self.assertGreaterEqual(a, GO.A_SOFT - 1e-9)
    self.assertLessEqual(a, 1.5)
    # incoming plan already braking harder: pass through (never weaken)
    on, a = self.step(rl(d=10.0, v=v - 1.0, a=0.0), v_ego=v, a_t=-1.5, a_prev=-1.5)
    self.assertFalse(on)
    self.assertEqual(a, -1.5)

  def test_cap_decel_stays_near_soft_bound(self):
    """A large catch-up error must not slam into A_FLOOR; peak lift stays near A_SOFT."""
    v = 20 * MPH
    des = GO.desired_follow_gap(v, self.p)
    # ego 4 mph faster than the lead, gap well short, plan asking for +1.5
    on, a = self.arm_and_roll(t=2.5, v1=v, d=0.70 * des, v_lead=v - 4.0 * MPH, a_lead=0.0, a_t=1.5)
    self.assertTrue(on)
    # after the jerk ramp-in, settle at the soft ease-off, well above the floor
    for _ in range(int(1.5 / DT)):
      on, a = self.step(rl(d=0.70 * des, v=v - 4.0 * MPH, a=0.0), v_ego=v, a_t=1.5, a_prev=a)
    self.assertTrue(on)
    self.assertGreaterEqual(a, GO.A_SOFT - 1e-9)
    self.assertLess(a, 0.0)
    self.assertGreater(a, GO.A_FLOOR + 0.10)  # must not be using the -0.3 floor
    self.assertAlmostEqual(a, GO.A_SOFT, places=2)

  def test_smooth_release_as_gap_closes(self):
    """As the gap walks from SHORT_FRAC to 100%, the cap blends toward the plan
    monotonically and turns off at the target; no abrupt jump off the floor."""
    v = 20 * MPH
    des = GO.desired_follow_gap(v, self.p)
    on, a = self.arm_and_roll(t=2.0, v1=v, d=GO.SHORT_FRAC * des, v_lead=v, a_lead=0.0, a_t=1.5)
    self.assertTrue(on)
    # settle at full short-gap cap
    for _ in range(int(1.5 / DT)):
      on, a = self.step(rl(d=GO.SHORT_FRAC * des, v=v, a=0.0), v_ego=v, a_t=1.5, a_prev=a)
    self.assertTrue(on)
    a_short = a
    self.assertGreaterEqual(a_short, GO.A_SOFT - 1e-9)
    # walk the gap up in small steps; commanded accel must rise toward the plan
    accels = [a_short]
    gaps = []
    n = 20
    for i in range(1, n + 1):
      frac = GO.SHORT_FRAC + (1.0 - GO.SHORT_FRAC) * (i / n)
      d = frac * des
      on, a = self.step(rl(d=d, v=v, a=0.0), v_ego=v, a_t=1.5, a_prev=accels[-1])
      accels.append(a)
      gaps.append(frac)
    # monotonic: each step is at least as close to the plan as the last
    for i in range(1, len(accels)):
      self.assertGreaterEqual(accels[i], accels[i - 1] - 1e-9)
      self.assertLessEqual(accels[i] - accels[i - 1], GO.JERK_UP * DT + 1e-9)
    # finish the jerk-limited release at/over the target (may take a few steps)
    for _ in range(int(3.0 / DT)):
      on, a = self.step(rl(d=des + 0.05, v=v, a=0.0), v_ego=v, a_t=1.5, a_prev=a)
      accels.append(a)
      if not on:
        break
    self.assertFalse(on)
    self.assertEqual(a, 1.5)
    # mid-window (around 95%) is strictly milder than the well-short cap, and
    # not a jump all the way to the plan in one sample
    mid_a = accels[n // 2]
    self.assertGreater(mid_a, a_short + 0.05)
    self.assertLess(mid_a, 1.5 - 0.05)

  def test_no_cap_when_already_opening_or_at_target(self):
    v = 20 * MPH
    des = GO.desired_follow_gap(v, self.p)
    # gap already at the target, ego matching lead: no cap
    on, a = self.arm_and_roll(t=1.5, v1=v, d=des + 0.5, v_lead=v, a_lead=0.0, a_t=1.2)
    self.assertFalse(on); self.assertEqual(a, 1.2)
    # short gap but already sitting V_OPEN under the lead: no cap (don't hold back)
    self.go.reset()
    on, a = self.arm_and_roll(t=1.5, v1=v, d=0.8 * des, v_lead=v + GO.V_OPEN + 0.5, a_lead=0.2, a_t=1.2)
    self.assertFalse(on); self.assertEqual(a, 1.2)

  def test_no_cap_when_lead_brakes_or_fcw(self):
    v = 20 * MPH
    self.arm_and_roll(t=1.5, v1=v, d=13.0, v_lead=v, a_lead=0.0, a_t=1.2)
    on, a = self.step(rl(d=13.0, v=v, a=-0.8), v_ego=v, a_t=0.4, a_prev=0.4)
    self.assertFalse(on); self.assertEqual(a, 0.4)
    on, a = self.step(rl(d=13.0, v=v, a=0.0), v_ego=v, fcw=True, a_t=0.4, a_prev=0.4)
    self.assertFalse(on); self.assertEqual(a, 0.4)

  def test_expires_at_end_speed_and_episode_time(self):
    v = GO.END_V + 0.05
    on, a = self.arm_and_roll(t=1.5, v1=v, d=12.0, v_lead=v, a_lead=0.0, a_t=1.0)
    self.assertFalse(on); self.assertEqual(a, 1.0)
    self.go.reset()
    self.step(rl(d=9.05), v_ego=0.0, standstill=True, a_t=-0.5, a_prev=-0.5)
    # run past MAX_EPISODE_S at a speed where the cap would otherwise apply
    ev = 20 * MPH
    steps = int((GO.MAX_EPISODE_S + 0.5) / DT)
    last = (False, 1.2)
    for i in range(steps):
      last = self.step(rl(d=12.0, v=ev, a=0.0), v_ego=ev, a_t=1.2, a_prev=1.2)
    self.assertFalse(last[0]); self.assertEqual(last[1], 1.2)

  def test_jerk_limited(self):
    v = 20 * MPH
    self.step(rl(d=9.05), v_ego=0.0, standstill=True, a_t=-0.5, a_prev=-0.5)
    a_prev = 1.5
    lows = []
    for i in range(int(1.5 / DT)):
      on, a = self.step(rl(d=12.0, v=v - 0.5, a=0.0), v_ego=v, a_t=1.5, a_prev=a_prev)
      if on:
        lows.append(a_prev - a)
        a_prev = a
    self.assertTrue(lows)
    self.assertLessEqual(max(lows), GO.JERK_DOWN * DT + 1e-9)

  def test_not_armed_without_standstill(self):
    # already moving: must not start an episode (steady follow unchanged)
    v = 25 * MPH
    des = GO.desired_follow_gap(v, self.p)
    on, a = self.step(rl(d=0.7 * des, v=v, a=0.0), v_ego=v, a_t=1.0, a_prev=1.0)
    self.assertFalse(on); self.assertEqual(a, 1.0)
    for _ in range(int(2.0 / DT)):
      on, a = self.step(rl(d=0.7 * des, v=v, a=0.0), v_ego=v, a_t=1.0, a_prev=1.0)
    self.assertFalse(on); self.assertEqual(a, 1.0)

  def test_disabled(self):
    GO.ENABLED = False
    try:
      on, a = self.arm_and_roll(t=1.5, v1=20 * MPH, d=12.0, v_lead=20 * MPH, a_t=1.2)
      self.assertFalse(on); self.assertEqual(a, 1.2)
    finally:
      GO.ENABLED = True

  def test_fuzz(self):
    r = np.random.default_rng(21)
    from openpilot.cereal import log
    personalities = (log.LongitudinalPersonality.aggressive, log.LongitudinalPersonality.standard,
                     log.LongitudinalPersonality.relaxed)
    for _ in range(20000):
      if r.random() < 0.03:
        self.go.reset()
      vals = [r.normal(12, 8), r.normal(0, 4), float("nan"), float("inf")]
      ld = rl(d=float(r.choice(vals)) if r.random() < 0.05 else float(r.uniform(0, 40)),
              v=float(r.uniform(-1, 30)), a=float(r.uniform(-5, 3)), present=bool(r.random() < 0.95))
      v = float(r.uniform(0, 20)); a_t = float(r.uniform(-3.5, 2)); a_p = float(r.uniform(-3.5, 2))
      on, a = self.step(ld, v_ego=v, standstill=v < 0.01, fcw=bool(r.random() < 0.05),
                        a_t=a_t, a_prev=a_p, lock=bool(r.random() < 0.2), boost=bool(r.random() < 0.2),
                        personality=personalities[int(r.integers(0, 3))])
      self.assertTrue(math.isfinite(a))
      self.assertLessEqual(a, a_t + 1e-9)  # never raises the plan
      if not on:
        self.assertEqual(a, a_t)
      else:
        # when we cap, we never go below A_SOFT unless the plan already did
        self.assertGreaterEqual(a, min(a_t, GO.A_SOFT) - 1e-9)
        self.assertGreaterEqual(a, min(a_t, GO.A_FLOOR) - 1e-9)
        self.assertLess(a, a_t)  # actually capped



if __name__ == "__main__":
  unittest.main()
