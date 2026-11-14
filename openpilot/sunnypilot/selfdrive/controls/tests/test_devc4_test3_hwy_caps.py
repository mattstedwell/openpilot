"""t312u: Agg/Std positive accel caps equal Relaxed's at highway speed."""
import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install
_install()

from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel  # noqa: E402

MPH = 0.44704


@pytest.mark.parametrize("mph", [45, 50, 55, 65, 75, 85])
@pytest.mark.parametrize("boost", [False, True])
def test_agg_std_equal_relaxed_at_highway(mph, boost):
  v = mph * MPH
  assert get_max_accel(v, boost, False) == pytest.approx(get_max_accel(v, False, True))


def test_blend_is_continuous_and_between():
  vs = np.linspace(38 * MPH, 47 * MPH, 400)
  caps = [get_max_accel(v, False, False) for v in vs]
  assert max(abs(b - a) for a, b in zip(caps, caps[1:], strict=False)) < 0.01  # no step
  for v, c in zip(vs, caps, strict=True):
    assert get_max_accel(v, False, True) - 1e-9 <= c


@pytest.mark.parametrize("mph", [0, 10, 20, 30, 39])
def test_below_40mph_unchanged(mph):
  v = mph * MPH
  stock = float(np.interp(v, [0., 10., 25., 40.], [1.6, 1.2, 0.8, 0.6]))
  assert get_max_accel(v, False, False) == pytest.approx(stock)
  assert get_max_accel(v, False, True) == pytest.approx(0.8 * stock)
