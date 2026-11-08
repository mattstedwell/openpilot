import numpy as np
import pytest
from types import SimpleNamespace

from cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel, limit_accel_in_turns
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc, CRUISE_MAX_ACCEL, N, COST_DIM, PARAM_DIM,
)


def test_launch_accel_curve():
  assert get_max_accel(0., True) == pytest.approx(2.)
  assert get_max_accel(5., True) == pytest.approx(1.6)
  assert get_max_accel(10., True) == pytest.approx(1.2)
  for speed in np.linspace(10., 45., 40):
    assert get_max_accel(speed, True) == get_max_accel(speed)
  assert get_max_accel(0.) == 1.6
  assert get_max_accel(25.) == .8


def test_turn_budget_still_limits_launch():
  cp = SimpleNamespace(steerRatio=15., wheelbase=2.7)
  # 2.0 is the requested ceiling, NOT permission to bypass the existing
  # combined lateral/longitudinal budget (1.7 near standstill).
  clipped = limit_accel_in_turns(0., 0., [-3.5, get_max_accel(0., True)], cp)
  assert clipped[1] == pytest.approx(1.7)
  assert clipped[0] == -3.5
  assert limit_accel_in_turns(5., 100., [-3.5, 2.], cp)[1] < 1.7


def test_mpc_cruise_obstacle_accepts_launch_budget():
  mpc = LongitudinalMpc.__new__(LongitudinalMpc)
  mpc.x0 = np.zeros(3)
  mpc.yref = np.zeros((N + 1, COST_DIM))
  mpc.params = np.zeros((N + 1, PARAM_DIM))
  mpc.a_prev = np.zeros(N + 1)
  mpc.x_sol = np.zeros((N + 1, 3))
  mpc.crash_cnt = 0
  mpc.solver = SimpleNamespace(set=lambda *args: None)
  mpc.run = lambda: None
  absent = SimpleNamespace(status=False, modelProb=0.)
  radar = SimpleNamespace(leadOne=absent, leadTwo=absent)
  mpc.process_lead = lambda l: np.column_stack((np.full(N + 1, 1000.), np.zeros(N + 1)))
  mpc.update(radar, 20., log.LongitudinalPersonality.aggressive)
  baseline = mpc.params.copy()
  mpc.update(radar, 20., log.LongitudinalPersonality.aggressive, cruise_max_accel=1.7)
  assert np.all(mpc.params[1:, 2] > baseline[1:, 2])
  np.testing.assert_array_equal(mpc.params[:, [0, 1, 4, 5]], baseline[:, [0, 1, 4, 5]])
  mpc.update(radar, 20., log.LongitudinalPersonality.aggressive, cruise_max_accel=CRUISE_MAX_ACCEL)
  np.testing.assert_array_equal(mpc.params, baseline)
