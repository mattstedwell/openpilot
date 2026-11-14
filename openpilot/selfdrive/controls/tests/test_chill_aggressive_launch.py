from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel, get_cruise_accel
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.sunnypilot.selfdrive.car.chill_aggressive import ChillAggressiveLaunch


def test_launch_accel_curve():
  # t314u: Aggressive's extra low-speed accel is part of its own smooth
  # curve (keyed on personality, not the launch flag), so there is no step
  # when the old launch boost used to arm or disarm. The boost helper
  # still exists but the planner no longer adds it.
  assert ACCEL_MAX == pytest.approx(2.0)
  assert ChillAggressiveLaunch.accel_extra(0.) == pytest.approx(0.4)
  assert ChillAggressiveLaunch.accel_extra(10.) == 0.
  assert get_max_accel(0., True) == pytest.approx(1.55)
  assert get_max_accel(5., True) == pytest.approx(1.3339, abs=1e-3)
  assert get_max_accel(10., True) == pytest.approx(1.0689, abs=1e-3)
  assert get_max_accel(0.) == pytest.approx(1.25)
  # Agg/Std == Relaxed at highway speed (25 m/s ~ 56 mph).
  assert get_max_accel(25.) == pytest.approx(get_max_accel(25., True))
  assert get_max_accel(25.) == pytest.approx(0.5291, abs=1e-3)
  for speed in np.linspace(0., 22., 23):
    assert get_max_accel(speed, True) >= get_max_accel(speed) - 1e-9
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
