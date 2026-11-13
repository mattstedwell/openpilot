from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

import pytest

from openpilot.cereal import log
from openpilot.sunnypilot.selfdrive.car.chill_relaxed import is_chill_relaxed
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_T_FOLLOW, get_jerk_factor, get_lead_stop_extra
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel
from opendbc.car.interfaces import ACCEL_MAX

P = log.LongitudinalPersonality


def test_profile_gating():
  for experimental in (False, True):
    for personality in P.schema.enumerants.values():
      assert is_chill_relaxed(experimental, personality) == (not experimental and personality == P.relaxed)


def test_t_follow():
  # Stop-and-go unchanged; highway gap grows. Same as C3X.
  assert get_T_FOLLOW(P.relaxed, False, 0.) == pytest.approx(1.75)
  assert get_T_FOLLOW(P.relaxed, False, 13.4) == pytest.approx(1.75)
  assert get_T_FOLLOW(P.relaxed, False, 30.) == pytest.approx(2.15)
  for v in (0., 30.):
    assert get_T_FOLLOW(P.relaxed, True, v) == 1.75
    assert get_T_FOLLOW(P.standard, False, v) == 1.45
    assert get_T_FOLLOW(P.aggressive, False, v) == 1.25


def test_jerk_and_accel():
  assert get_jerk_factor(P.relaxed, False) == pytest.approx(1.3)
  assert get_jerk_factor(P.relaxed, True) == 1.0
  assert get_jerk_factor(P.standard, False) == 1.0
  assert get_jerk_factor(P.aggressive, False) == 0.5
  # This branch pins Standard to the C3X stock curve; Relaxed is 80% of it.
  # Official unpatched sunnypilot/dev is still a flat ACCEL_MAX = 2.0.
  assert ACCEL_MAX == pytest.approx(2.0)
  assert get_max_accel(0., chill_relaxed_mode=True) == pytest.approx(1.28)
  assert get_max_accel(25., chill_relaxed_mode=True) == pytest.approx(0.64)
  assert get_max_accel(40., chill_relaxed_mode=True) == pytest.approx(0.48)
  assert get_max_accel(0.) == pytest.approx(1.6)
  assert get_max_accel(25.) == pytest.approx(0.8)


def test_standstill_gaps_unchanged():
  ft = 0.3048
  assert get_lead_stop_extra(P.relaxed, False) == pytest.approx(10 * ft)
  assert get_lead_stop_extra(P.standard, False) == pytest.approx(5 * ft)
  assert get_lead_stop_extra(P.aggressive, False) == pytest.approx(5 * ft)
  assert get_lead_stop_extra(P.aggressive, True) == 0.0
