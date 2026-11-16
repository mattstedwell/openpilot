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
  """Post-launch soft accel cap: open a short follow gap a little faster, comfortably."""

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
    """Standstill arm, then roll past 0.9 s and 6.5 mph so the cap may apply."""
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
    # floor stays a light ease-off; 2 mph under is a setpoint, not a harder brake
    self.assertAlmostEqual(GO.A_FLOOR, -0.30, places=2)

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

  def test_inactive_at_standstill_and_first_motion(self):
    on, a = self.step(rl(d=9.05), v_ego=0.0, standstill=True, a_t=-0.5, a_prev=-0.5)
    self.assertFalse(on); self.assertEqual(a, -0.5)
    # 0.5 s later, still below lock end and MIN_MOTION_S: unchanged
    on, a = self.step(rl(d=10.0, v=3.0, a=1.0), v_ego=1.0, a_t=1.5, a_prev=-0.5)
    self.assertFalse(on); self.assertEqual(a, 1.5)
    # 0.8 s total, past 6.5 mph would still be too early (MIN_MOTION_S=0.9)
    for _ in range(6):
      on, a = self.step(rl(d=12.0, v=8.0, a=0.5), v_ego=8.0, a_t=1.5, a_prev=1.5)
    self.assertFalse(on); self.assertEqual(a, 1.5)

  def test_inactive_while_lock_or_boost_active(self):
    self.step(rl(d=9.05), v_ego=0.0, standstill=True, a_t=-0.5, a_prev=-0.5)
    for _ in range(int(1.2 / DT)):
      on, a = self.step(rl(d=12.0, v=8.0, a=0.5), v_ego=8.0, a_t=1.5, a_prev=1.5, lock=True)
    self.assertFalse(on); self.assertEqual(a, 1.5)
    self.go.reset()
    self.step(rl(d=9.05), v_ego=0.0, standstill=True, a_t=-0.5, a_prev=-0.5)
    for _ in range(int(1.2 / DT)):
      on, a = self.step(rl(d=12.0, v=8.0, a=0.5), v_ego=8.0, a_t=1.5, a_prev=1.5, boost=True)
    self.assertFalse(on); self.assertEqual(a, 1.5)

  def test_inactive_below_lock_end_speed(self):
    self.step(rl(d=9.05), v_ego=0.0, standstill=True, a_t=-0.5, a_prev=-0.5)
    v = GO.LOCK_END_V - 0.05
    for _ in range(int(1.2 / DT)):
      on, a = self.step(rl(d=10.0, v=v + 1.0, a=0.5), v_ego=v, a_t=1.5, a_prev=1.5)
    self.assertFalse(on); self.assertEqual(a, 1.5)

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
    # ego faster than lead, short gap: cap applies, not below floor
    self.assertTrue(on)
    self.assertGreaterEqual(a, GO.A_FLOOR - 1e-9)
    self.assertLessEqual(a, 1.5)
    # incoming plan already braking harder: pass through (never weaken)
    on, a = self.step(rl(d=10.0, v=v - 1.0, a=0.0), v_ego=v, a_t=-1.5, a_prev=-1.5)
    self.assertFalse(on)
    self.assertEqual(a, -1.5)

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
        # when we cap, we never go below the floor unless the plan already did
        self.assertGreaterEqual(a, min(a_t, GO.A_FLOOR) - 1e-9)
        self.assertLess(a, a_t)  # actually capped



if __name__ == "__main__":
  unittest.main()
