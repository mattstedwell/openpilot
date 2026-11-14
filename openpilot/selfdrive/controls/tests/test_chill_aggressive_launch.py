from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel, get_cruise_accel
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.sunnypilot.selfdrive.car.chill_aggressive import ChillAggressiveLaunch


def test_launch_accel_curve():
  # C3X / old test3: Standard is the [1.6, 1.2, 0.8, 0.6] curve; launch is
  # that curve + 0.4*(1-v/10), so 2.0 at a stop, 1.6 at 5 m/s, 1.2 at 10 m/s.
  # The boost is no longer a no-op (it was when Standard was a flat 2.0).
  assert ACCEL_MAX == pytest.approx(2.0)
  assert ChillAggressiveLaunch.accel_extra(0.) == pytest.approx(0.4)
  assert ChillAggressiveLaunch.accel_extra(5.) == pytest.approx(0.2)
  assert ChillAggressiveLaunch.accel_extra(10.) == 0.
  assert get_max_accel(0., True) == pytest.approx(2.0)
  assert get_max_accel(5., True) == pytest.approx(1.6)
  assert get_max_accel(10., True) == pytest.approx(1.2)
  for speed in np.linspace(10., 45., 30):
    assert get_max_accel(speed, True) == pytest.approx(get_max_accel(speed))
  assert get_max_accel(0.) == pytest.approx(1.6)
  # t312u: Standard == Relaxed at highway speed (25 m/s ~ 56 mph).
  assert get_max_accel(25.) == pytest.approx(0.64)
  # Below 10 m/s, launch sits above Standard.
  assert get_max_accel(0., True) > get_max_accel(0.)
  assert get_max_accel(5., True) > get_max_accel(5.)


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
