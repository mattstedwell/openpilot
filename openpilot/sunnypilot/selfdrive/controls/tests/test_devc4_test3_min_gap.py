"""devc4-test3 hard min-gap floor and Aggressive T_FOLLOW 1.5 s."""
from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

import numpy as np
import pytest
from types import SimpleNamespace

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  get_T_FOLLOW, STOP_DISTANCE, LEAD_DANGER_FACTOR,
)
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_min_gap import (
  get_min_gap_floor, desired_bumper_gap, danger_factor_for_floor,
  apply_min_gap_floor, FLOOR_RESTORE_A, FT,
)

P = log.LongitudinalPersonality
MPH = 0.44704


def test_floor_is_planned_standstill_gap():
  assert get_min_gap_floor(P.aggressive, False) == pytest.approx(7.524)
  assert get_min_gap_floor(P.standard, False) == pytest.approx(7.524)
  assert get_min_gap_floor(P.relaxed, False) == pytest.approx(9.048)
  assert get_min_gap_floor(P.aggressive, False) / FT == pytest.approx(24.69, abs=0.01)
  assert get_min_gap_floor(P.relaxed, False) / FT == pytest.approx(29.69, abs=0.01)
  # Experimental Aggressive has no +5 ft extra
  assert get_min_gap_floor(P.aggressive, True) == pytest.approx(STOP_DISTANCE)


def test_agg_t_follow_chill_is_1_5():
  for v in (0.0, 5 * MPH, 15 * MPH, 30 * MPH, 45 * MPH):
    assert get_T_FOLLOW(P.aggressive, False, v) == 1.5
    assert get_T_FOLLOW(P.aggressive, True, v) == 1.25
    assert get_T_FOLLOW(P.standard, False, v) == 1.45


def test_desired_bumper_at_city_speeds():
  # At 30 mph, extras are blended out: T*v + STOP.
  v = 30 * MPH
  d = desired_bumper_gap(v, v, P.aggressive, False)
  assert d == pytest.approx(1.5 * v + 6.0)
  assert d / FT == pytest.approx(85.7, abs=0.2)
  # Time-gap (bumper / v) is T + STOP/v, not T itself.
  assert (d / v) == pytest.approx(1.5 + 6.0 / v)


def test_danger_factor_sits_on_floor_at_standstill():
  f = danger_factor_for_floor(0.0, 0.0, P.aggressive, False)
  assert f == pytest.approx(1.0)
  # At 30 mph, official 0.75 already clears the floor.
  f30 = danger_factor_for_floor(30 * MPH, 30 * MPH, P.aggressive, False)
  assert f30 == pytest.approx(LEAD_DANGER_FACTOR)


def test_pass_through_at_or_above_floor():
  floor = 7.524
  a, ss, iv = apply_min_gap_floor(0.5, False, floor, 13.4, 13.4, 0.5, floor)
  assert a == 0.5 and ss is False and iv is False
  a, ss, iv = apply_min_gap_floor(0.5, False, floor + 1.0, 13.4, 13.4, 0.5, floor)
  assert iv is False and a == 0.5


def test_never_positive_inside_floor():
  floor = 7.524
  a, ss, iv = apply_min_gap_floor(1.2, False, 5.0, 13.4, 13.4, 1.2, floor, dt=0.05)
  assert iv is True
  assert a < 0.0


def test_brakes_to_restore_inside_floor():
  floor = 7.524
  a, ss, iv = apply_min_gap_floor(0.0, False, 5.0, 5.0, 5.0, 0.0, floor, dt=0.05)
  assert iv is True
  assert a < 0.0  # first frame starts braking (jerk-limited toward restore)
  # After enough frames we should be at a negative restore target
  prev = a
  for _ in range(20):
    a, ss, iv = apply_min_gap_floor(0.0, False, 5.0, 5.0, 5.0, prev, floor, dt=0.05)
    prev = a
  assert a < -0.4


def test_cut_in_inside_floor_brakes_immediately():
  """A lead that appears at 15 ft (~4.57 m) while at 30 mph is inside
  the 24.69 ft floor. First frame must not be positive and must start
  braking. Instant restore is physically impossible."""
  floor = 7.524
  d = 15 * 0.3048
  a, ss, iv = apply_min_gap_floor(0.3, False, d, 30 * MPH, 30 * MPH, 0.3, floor, dt=0.05)
  assert iv is True
  assert a < 0.0
  # Closing fast (cut-in with us faster) is urgent: at least restore.
  a2, _, _ = apply_min_gap_floor(0.3, False, d, 30 * MPH, 25 * MPH, 0.3, floor, dt=0.05)
  assert a2 < 0.0


def test_should_stop_when_slow_and_inside_floor():
  floor = 7.524
  a, ss, iv = apply_min_gap_floor(-0.2, False, 5.0, 0.1, 0.0, -0.2, floor)
  assert iv is True and ss is True


def test_nonfinite_or_no_floor_passes_through():
  a, ss, iv = apply_min_gap_floor(0.4, False, float("nan"), 5.0, 5.0, 0.4, 7.524)
  assert iv is False and a == 0.4
  a, ss, iv = apply_min_gap_floor(0.4, True, 5.0, 5.0, 5.0, 0.4, None)
  assert iv is False and a == 0.4 and ss is True
