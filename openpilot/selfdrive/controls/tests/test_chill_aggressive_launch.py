from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel, get_cruise_accel
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.sunnypilot.selfdrive.car.chill_aggressive import ChillAggressiveLaunch


def test_launch_accel_curve():
  # Official stock is a flat 2.0. Launch is stock + 0.4*(1-v/10), clipped
  # at ACCEL_MAX, so the boost is a no-op at the actuator on this tree.
  # C3X: 2.0 at 0 (1.6+0.4), 1.6 at 5, 1.2 at 10, and stock 1.6 / 0.8 at 0 / 25.
  assert ACCEL_MAX == pytest.approx(2.0)
  assert ChillAggressiveLaunch.accel_extra(0.) == pytest.approx(0.4)
  assert ChillAggressiveLaunch.accel_extra(5.) == pytest.approx(0.2)
  assert ChillAggressiveLaunch.accel_extra(10.) == 0.
  assert get_max_accel(0., True) == pytest.approx(2.0)
  assert get_max_accel(5., True) == pytest.approx(2.0)
  assert get_max_accel(10., True) == pytest.approx(2.0)
  for speed in np.linspace(0., 45., 40):
    assert get_max_accel(speed, True) == pytest.approx(get_max_accel(speed))
    assert get_max_accel(speed) == pytest.approx(2.0)


def test_get_cruise_accel_respects_personality_cap():
  # Official sunnypilot/dev has no cruise obstacle in the MPC; the
  # personality cap is applied to the get_cruise_accel candidate.
  # A large speed deficit requests the cap, then jerk-limits.
  a = get_cruise_accel(v_cruise=30.0, v_ego=0.0, a_cruise_prev=0.0, dt=0.05, cap=1.6)
  assert a <= 1.6 + 1e-9
  a2 = get_cruise_accel(v_cruise=30.0, v_ego=0.0, a_cruise_prev=0.0, dt=0.05, cap=None)
  # Official jerk at v=0 is 1.6 m/s³, so first 0.05 s step is 0.08, under 2.0.
  assert a2 == pytest.approx(min(2.0, 1.6 * 0.05))
  assert a == pytest.approx(min(1.6, 1.6 * 0.05))
