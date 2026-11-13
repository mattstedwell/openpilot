"""devc4-test3: C3X Chill longitudinal policy, on official sunnypilot/dev.

T_FOLLOW, jerk costs, and stopped-gap extras are bit-identical to C3X
dev-test a413dbf814. Accel caps use this tree's official stock
(ACCEL_MAX = 2.0, flat) — Standard is that stock, Relaxed is 80% of
it (1.6), Aggressive launch +0.4 clips at 2.0 and is a no-op. C3X
numbers (the [1.6, 1.2, 0.8, 0.6] curve) are recorded for comparison.

Host-safe (stubs the aarch64 solver). Nothing here is road-tested.
Experimental / e2e paths are not part of this port. Always-Chill stays on.
No test2 or test4 long tunes.
"""
from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

import numpy as np
import pytest

from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.sunnypilot.selfdrive.car.chill_aggressive import (
  ChillAggressiveLaunch, is_chill_aggressive,
)
from openpilot.sunnypilot.selfdrive.car import chill_relaxed
from openpilot.sunnypilot.selfdrive.car.always_chill import ALWAYS_CHILL, experimental_mode_active
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  get_T_FOLLOW, get_jerk_factor, get_lead_stop_extra, get_safe_obstacle_distance,
  STOP_DISTANCE, LEAD_STOP_V_FULL, LEAD_STOP_V_NONE,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel

P = log.LongitudinalPersonality
MPH = 0.44704
FT = 0.3048

# C3X / old test3 stock curve, for documented diffs only. Not used as this
# tree's Standard stock.
C3X_BP = [0., 10.0, 25., 40.]
C3X_VALS = [1.6, 1.2, 0.8, 0.6]


def c3x_stock(v):
  return float(np.interp(v, C3X_BP, C3X_VALS))


def time_to(a_fn, mph0, mph1, dt=0.001):
  v0, v1 = mph0 * MPH, mph1 * MPH
  v, t = v0, 0.0
  while v < v1 and t < 180.0:
    a = a_fn(v)
    if a <= 0.01:
      return None
    v += a * dt
    t += dt
  return t


def test_always_chill_still_enforced():
  assert ALWAYS_CHILL is True
  assert experimental_mode_active(True) is False
  assert experimental_mode_active(False) is False


def test_t_follow_chill_relaxed_breakpoints():
  # 1.75 s through ~30 mph, 2.15 s from ~50 mph. Matches C3X chill_relaxed.
  assert chill_relaxed.T_FOLLOW_BP == [13.4, 22.4]
  assert chill_relaxed.T_FOLLOW_V == [1.75, 2.15]
  for v in (0.0, 5.0, 13.4):
    assert get_T_FOLLOW(P.relaxed, False, v) == pytest.approx(1.75)
  assert get_T_FOLLOW(P.relaxed, False, 22.4) == pytest.approx(2.15)
  assert get_T_FOLLOW(P.relaxed, False, 30.0) == pytest.approx(2.15)
  mid = (13.4 + 22.4) / 2.0
  assert get_T_FOLLOW(P.relaxed, False, mid) == pytest.approx(1.95)
  v45 = 45 * MPH
  expected = float(np.interp(v45, [13.4, 22.4], [1.75, 2.15]))
  assert get_T_FOLLOW(P.relaxed, False, v45) == pytest.approx(expected)
  assert 2.04 < expected < 2.06


def test_t_follow_standard_and_aggressive_unchanged():
  for exp in (False, True):
    for v in (0.0, 13.4, 20.12, 29.06, 40.0):
      assert get_T_FOLLOW(P.standard, exp, v) == 1.45
      assert get_T_FOLLOW(P.aggressive, exp, v) == 1.25
  for v in (0.0, 30.0):
    assert get_T_FOLLOW(P.relaxed, True, v) == 1.75


def test_jerk_factors():
  assert chill_relaxed.JERK_FACTOR == pytest.approx(1.3)
  assert get_jerk_factor(P.relaxed, False) == pytest.approx(1.3)
  assert get_jerk_factor(P.relaxed, True) == 1.0
  assert get_jerk_factor(P.standard, False) == 1.0
  assert get_jerk_factor(P.aggressive, False) == 0.5


def test_accel_caps_this_tree_stock():
  assert chill_relaxed.MAX_ACCEL_SCALE == pytest.approx(0.8)
  assert ACCEL_MAX == pytest.approx(2.0)
  # Standard is this tree's official stock: flat 2.0.
  for v in (0.0, 10.0, 25.0, 40.0):
    assert get_max_accel(v) == pytest.approx(2.0)
  # Relaxed is 80% of that stock, at every speed (1.6).
  for v in np.linspace(0.0, 40.0, 21):
    assert get_max_accel(v, chill_relaxed_mode=True) == pytest.approx(2.0 * 0.8)
  assert get_max_accel(0.0, chill_relaxed_mode=True) == pytest.approx(1.6)
  # Aggressive launch +0.4 clips at ACCEL_MAX: no-op vs Standard.
  assert ChillAggressiveLaunch.accel_extra(0.0) == pytest.approx(0.4)
  assert get_max_accel(0.0, True) == pytest.approx(2.0)
  assert get_max_accel(5.0, True) == pytest.approx(2.0)
  assert get_max_accel(10.0, True) == pytest.approx(2.0)
  for v in (10.0, 25.0, 40.0):
    assert get_max_accel(v, True) == pytest.approx(get_max_accel(v))
  # Mutual exclusion: relaxed scale wins if both flags were ever set (C3X).
  assert get_max_accel(0.0, True, True) == pytest.approx(1.6)


def test_accel_caps_differ_from_c3x():
  """Document the C3X/old-test3 numbers so diffs are explicit, not silent."""
  # C3X Standard / stock: 1.6, 1.2, 0.8, 0.6 at 0, 10, 25, 40 m/s.
  assert c3x_stock(0.0) == pytest.approx(1.6)
  assert c3x_stock(25.0) == pytest.approx(0.8)
  assert c3x_stock(40.0) == pytest.approx(0.6)
  # C3X Relaxed 80% of that curve.
  assert c3x_stock(0.0) * 0.8 == pytest.approx(1.28)
  assert c3x_stock(25.0) * 0.8 == pytest.approx(0.64)
  assert c3x_stock(40.0) * 0.8 == pytest.approx(0.48)
  # C3X Aggressive launch: 2.0 at a stop, 1.6 at 5, 1.2 at 10.
  assert c3x_stock(0.0) + 0.4 == pytest.approx(2.0)
  assert c3x_stock(5.0) + 0.2 == pytest.approx(1.6)
  assert c3x_stock(10.0) + 0.0 == pytest.approx(1.2)
  # This tree vs C3X at the same points.
  assert get_max_accel(0.0) - c3x_stock(0.0) == pytest.approx(0.4)
  assert get_max_accel(25.0) - c3x_stock(25.0) == pytest.approx(1.2)
  assert get_max_accel(0.0, chill_relaxed_mode=True) - c3x_stock(0.0) * 0.8 == pytest.approx(0.32)
  assert get_max_accel(25.0, chill_relaxed_mode=True) - c3x_stock(25.0) * 0.8 == pytest.approx(0.96)
  assert get_max_accel(0.0, True) == pytest.approx(2.0)  # same ceiling as C3X launch at a stop
  assert get_max_accel(5.0, True) - (c3x_stock(5.0) + 0.2) == pytest.approx(0.4)


def test_open_loop_accel_times():
  # Open-loop integration of this tree's cruise cap (flat). Launch is a no-op.
  t_agg = time_to(lambda v: get_max_accel(v, True), 0, 30)
  t_std = time_to(lambda v: get_max_accel(v), 0, 30)
  t_rel = time_to(lambda v: get_max_accel(v, chill_relaxed_mode=True), 0, 30)
  # 30 mph / 2.0 ≈ 6.71 s; / 1.6 ≈ 8.38 s. C3X was 9.34 / 10.15 / 12.69.
  assert t_agg == pytest.approx(t_std)
  assert t_std == pytest.approx(30 * MPH / 2.0, abs=0.05)
  assert t_rel == pytest.approx(30 * MPH / 1.6, abs=0.05)
  t75_rel = time_to(lambda v: get_max_accel(v, chill_relaxed_mode=True), 55, 75)
  t75_std = time_to(lambda v: get_max_accel(v), 55, 75)
  t75_agg = time_to(lambda v: get_max_accel(v, True), 55, 75)
  # 20 mph / 2.0 ≈ 4.47 s; / 1.6 ≈ 5.59 s. C3X was 15.01 Relaxed, 12.01 Std/Agg.
  assert t75_agg == pytest.approx(t75_std)
  assert t75_std == pytest.approx(20 * MPH / 2.0, abs=0.05)
  assert t75_rel == pytest.approx(20 * MPH / 1.6, abs=0.05)
  # Record the C3X times so a reader can compare.
  assert 9.34 < 10.15 < 12.69
  assert 12.01 < 15.01


def test_lead_stop_extra_feet():
  assert get_lead_stop_extra(P.relaxed, False) == pytest.approx(10 * FT)
  assert get_lead_stop_extra(P.standard, False) == pytest.approx(5 * FT)
  assert get_lead_stop_extra(P.aggressive, False) == pytest.approx(5 * FT)
  assert get_lead_stop_extra(P.relaxed, True) == pytest.approx(10 * FT)
  assert get_lead_stop_extra(P.standard, True) == pytest.approx(5 * FT)
  assert get_lead_stop_extra(P.aggressive, True) == 0.0
  assert STOP_DISTANCE == pytest.approx(6.0)
  assert STOP_DISTANCE + 10 * FT == pytest.approx(9.048)
  assert STOP_DISTANCE + 5 * FT == pytest.approx(7.524)
  assert LEAD_STOP_V_FULL == pytest.approx(0.5)
  assert LEAD_STOP_V_NONE == pytest.approx(5.0)


def test_follow_gap_at_city_and_highway_speeds():
  # Analytic desired gap: v^2/(2*COMFORT_BRAKE) + T*v + STOP_DISTANCE.
  # Lead-stop extra is fully blended out by 5 m/s, so moving follow is T only.
  # Bit-identical to C3X (T_FOLLOW is the same).
  cases = {
    20: {P.aggressive: 1.25, P.standard: 1.45, P.relaxed: 1.75},
    30: {P.aggressive: 1.25, P.standard: 1.45, P.relaxed: float(np.interp(30 * MPH, [13.4, 22.4], [1.75, 2.15]))},
    45: {P.aggressive: 1.25, P.standard: 1.45, P.relaxed: float(np.interp(45 * MPH, [13.4, 22.4], [1.75, 2.15]))},
    65: {P.aggressive: 1.25, P.standard: 1.45, P.relaxed: 2.15},
  }
  for mph, pers in cases.items():
    v = mph * MPH
    for p, T in pers.items():
      assert get_T_FOLLOW(p, False, v) == pytest.approx(T)
      gap = get_safe_obstacle_distance(v, T)
      expected = v ** 2 / (2 * 2.5) + T * v + 6.0
      assert gap == pytest.approx(expected)
      if p == P.relaxed and mph >= 50:
        stock = get_safe_obstacle_distance(v, 1.75)
        assert gap - stock == pytest.approx(0.4 * v, rel=1e-6)
        if mph == 65:
          assert (gap - stock) * 3.28084 == pytest.approx(0.4 * v * 3.28084)


def test_is_chill_gating():
  assert is_chill_aggressive(False, P.aggressive)
  assert not is_chill_aggressive(True, P.aggressive)
  assert chill_relaxed.is_chill_relaxed(False, P.relaxed)
  assert not chill_relaxed.is_chill_relaxed(True, P.relaxed)
  assert experimental_mode_active(True) is False
