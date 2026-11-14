"""devc4-test3: C3X Chill longitudinal policy, on official sunnypilot/dev.

T_FOLLOW, jerk costs and stopped-gap extras are pinned to C3X dev-test
a413dbf814 / old test3. Accel caps (t314u/t316u) are no longer the C3X
piecewise curve: each personality has a smooth, gentler PCHIP curve
(devc4_test3_accel_curve) and Agg/Std blend onto Relaxed over 35-50 mph.
Open-loop 0-30 times are now Agg 11.102 / Std 12.801 / Rel 15.111 s (C3X
was 9.342 / 10.149 / 12.686) and 55-75 is 18.272 s for all three.

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
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  get_max_accel, A_CRUISE_MAX_VALS, A_CRUISE_MAX_BP,
)

P = log.LongitudinalPersonality
MPH = 0.44704
FT = 0.3048

# This branch's Standard stock is the C3X / old-test3 curve.
assert A_CRUISE_MAX_BP == [0., 10.0, 25., 40.]
assert A_CRUISE_MAX_VALS == [1.6, 1.2, 0.8, 0.6]


def c3x_stock(v):
  return float(np.interp(v, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS))


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


def test_accel_caps_smooth_curve():
  assert chill_relaxed.MAX_ACCEL_SCALE == pytest.approx(0.8)
  # Official unpatched sunnypilot/dev is still a flat 2.0; we pin below it.
  assert ACCEL_MAX == pytest.approx(2.0)
  # t314u smooth curves (m/s, cap): gentler than C3X at every speed.
  assert get_max_accel(0.0) == pytest.approx(1.25)
  assert get_max_accel(0.0, True) == pytest.approx(1.55)
  assert get_max_accel(0.0, chill_relaxed_mode=True) == pytest.approx(1.05)
  assert get_max_accel(10.0) == pytest.approx(0.9533, abs=1e-3)
  assert get_max_accel(25.0) == pytest.approx(0.5291, abs=1e-3)
  assert get_max_accel(40.0) == pytest.approx(0.421, abs=1e-3)
  for v in np.linspace(0.0, 40.0, 81):
    assert get_max_accel(v, chill_relaxed_mode=True) <= get_max_accel(v) + 1e-9 <= get_max_accel(v, True) + 2e-9
    assert get_max_accel(v) <= c3x_stock(v) + 1e-9
    assert get_max_accel(v, chill_relaxed_mode=True) <= 0.8 * c3x_stock(v) + 1e-9
  # Highway: Agg / Std == Relaxed at >= 50 mph (blend 35-50 mph).
  for v in (50 * MPH, 25.0, 40.0):
    assert get_max_accel(v) == pytest.approx(get_max_accel(v, chill_relaxed_mode=True))
    assert get_max_accel(v, True) == pytest.approx(get_max_accel(v, chill_relaxed_mode=True))
  # Relaxed wins if both flags were ever set (C3X mutual exclusion).
  assert get_max_accel(0.0, True, True) == pytest.approx(1.05)


def test_open_loop_accel_times():
  # Open-loop integration of the cruise cap (t314u smooth curve).
  t_agg = time_to(lambda v: get_max_accel(v, True), 0, 30)
  t_std = time_to(lambda v: get_max_accel(v), 0, 30)
  t_rel = time_to(lambda v: get_max_accel(v, chill_relaxed_mode=True), 0, 30)
  assert t_agg == pytest.approx(11.102, abs=0.05)
  assert t_std == pytest.approx(12.801, abs=0.05)
  assert t_rel == pytest.approx(15.111, abs=0.05)
  assert t_agg < t_std < t_rel
  t75_rel = time_to(lambda v: get_max_accel(v, chill_relaxed_mode=True), 55, 75)
  t75_std = time_to(lambda v: get_max_accel(v), 55, 75)
  t75_agg = time_to(lambda v: get_max_accel(v, True), 55, 75)
  assert t75_agg == pytest.approx(t75_rel, abs=0.01)
  assert t75_std == pytest.approx(t75_rel, abs=0.01)
  assert t75_rel == pytest.approx(18.272, abs=0.05)

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
