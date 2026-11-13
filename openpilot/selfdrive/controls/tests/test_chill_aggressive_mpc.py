from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

import numpy as np
import pytest
from types import SimpleNamespace

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc, get_lead_stop_extra, get_T_FOLLOW, N, T_IDXS, COST_DIM, PARAM_DIM,
)


@pytest.mark.parametrize('experimental', [False, True])
@pytest.mark.parametrize('personality,expected_ft', [
  (log.LongitudinalPersonality.aggressive, 5),
  (log.LongitudinalPersonality.standard, 5),
  (log.LongitudinalPersonality.relaxed, 10),
])
def test_lead_stop_gap_profile(experimental, personality, expected_ft):
  if experimental and personality == log.LongitudinalPersonality.aggressive:
    expected_ft = 0
  assert get_lead_stop_extra(personality, experimental) == pytest.approx(expected_ft * .3048)


@pytest.mark.parametrize('lead_speed', [0., .5, 2., 5., 15.])
@pytest.mark.parametrize('lead_number', [1, 2])
def test_lead_stop_gap_blending(lead_speed, lead_number):
  # Exercise MPC obstacle construction without requiring a nonlinear solver run.
  mpc = LongitudinalMpc.__new__(LongitudinalMpc)
  mpc.x0 = np.array([0., 10., 0.])
  mpc.yref = np.zeros((N + 1, COST_DIM))
  mpc.params = np.zeros((N + 1, PARAM_DIM))
  mpc.a_prev = np.zeros(N + 1)
  mpc.x_sol = np.zeros((N + 1, 3))
  mpc.crash_cnt = 0
  mpc.solver = SimpleNamespace(set=lambda *args: None)
  mpc.run = lambda: None
  lead = SimpleNamespace(present=True, status=True, modelProb=0.)
  absent = SimpleNamespace(present=False, status=False, modelProb=0.)
  radar = SimpleNamespace(leadOne=lead if lead_number == 1 else absent, leadTwo=lead if lead_number == 2 else absent)
  mpc.process_lead = lambda l: np.column_stack((
    np.full(N + 1, 5. if (getattr(l, "present", False) or getattr(l, "status", False)) else 1000.),
    np.full(N + 1, lead_speed),
  ))
  mpc.update(radar, personality=log.LongitudinalPersonality.aggressive, experimental_mode=True)
  baseline = mpc.params[:, 2].copy()
  mpc.update(radar, personality=log.LongitudinalPersonality.aggressive, experimental_mode=False)
  expected_extra = np.interp(lead_speed, [.5, 5.], [5 * .3048, 0.])
  np.testing.assert_allclose(baseline - mpc.params[:, 2], expected_extra)
  assert get_T_FOLLOW(log.LongitudinalPersonality.aggressive) == 1.25
  assert len(T_IDXS) == N + 1
