"""t312u/t316u: Agg/Std positive accel caps equal Relaxed's at highway
speed, reached through a wide C2 (smootherstep) blend over 35-50 mph."""
import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install
_install()

from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel  # noqa: E402

MPH = 0.44704


@pytest.mark.parametrize("mph", [50, 55, 65, 75, 85])
@pytest.mark.parametrize("agg", [False, True])
def test_agg_std_equal_relaxed_at_highway(mph, agg):
  v = mph * MPH
  assert get_max_accel(v, agg, False) == pytest.approx(get_max_accel(v, False, True))


@pytest.mark.parametrize("agg", [False, True])
def test_blend_is_smooth_and_between(agg):
  vs = np.linspace(30 * MPH, 55 * MPH, 5001)
  caps = np.array([get_max_accel(v, agg, False) for v in vs])
  rel = np.array([get_max_accel(v, False, True) for v in vs])
  assert (rel - 1e-9 <= caps).all()
  d = np.diff(caps) / np.diff(vs)
  assert (d <= 1e-12).all()                 # monotone, never rises with speed
  assert np.abs(d).max() < 0.08             # gentle: < 0.08 (m/s²)/(m/s)
  assert np.abs(np.diff(d)).max() < 1e-3    # no kink (slope continuous)
  # while accelerating at 1 m/s² through the blend, the cap changes < 0.1 m/s³
  assert np.abs(d).max() * 1.0 < 0.1


@pytest.mark.parametrize("mph", [0, 10, 20, 30])
def test_below_blend_own_curve_gentler_than_c3x(mph):
  v = mph * MPH
  stock = float(np.interp(v, [0., 10., 25., 40.], [1.6, 1.2, 0.8, 0.6]))
  assert get_max_accel(v, False, False) <= stock + 1e-9
  assert get_max_accel(v, False, True) <= 0.8 * stock + 1e-9
  assert get_max_accel(v, True, False) <= stock + 0.4 * (1 - min(v, 10.) / 10.) + 1e-9
