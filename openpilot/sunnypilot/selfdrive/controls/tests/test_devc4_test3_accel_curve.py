"""t314u / t316u: smooth, gentler positive accel ceilings and the
planner's positive-accel rise limit. Host-only; nothing road-tested."""
from pathlib import Path

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install
_install()

from openpilot.sunnypilot.selfdrive.controls.lib import devc4_test3_accel_curve as AC  # noqa: E402
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel  # noqa: E402

MPH = 0.44704
PERS = ("aggressive", "standard", "relaxed")
HERE = Path(__file__).resolve().parent
PLANNER = HERE.parents[3] / "selfdrive/controls/lib/longitudinal_planner.py"


def c3x(v):
  return float(np.interp(v, [0., 10., 25., 40.], [1.6, 1.2, 0.8, 0.6]))


@pytest.mark.parametrize("p", PERS)
def test_monotone_c1_and_bounded(p):
  vs = np.linspace(0, 40, 40001)  # knots end at 90 mph (40.2 m/s); flat above
  c = np.array([AC.smooth_accel_cap(v, p) for v in vs])
  d = np.diff(c) / np.diff(vs)
  assert (d <= 1e-12).all()
  assert np.abs(np.diff(d)).max() < 1e-3          # slope continuous (no kinks)
  assert c.min() >= 0.35                           # never so low it can't hold speed/climb
  assert c.max() <= 2.0


def test_pchip_matches_scipy_when_available():
  interp = pytest.importorskip("scipy.interpolate")
  for knots in (AC._RELAXED, AC._STANDARD, AC._AGGRESSIVE):
    ref = interp.PchipInterpolator(AC._X, knots)
    own = AC._Pchip(AC._X, knots)
    for v in np.linspace(0, 40.2, 403):
      assert own(v) == pytest.approx(float(ref(v)), abs=1e-9)


def test_gentler_than_c3x_and_ordered():
  for v in np.linspace(0, 40, 161):
    agg, std, rel = (AC.smooth_accel_cap(v, p) for p in PERS)
    assert rel <= std + 1e-9 <= agg + 2e-9
    assert std <= c3x(v) + 1e-9
    assert rel <= 0.8 * c3x(v) + 1e-9
    assert agg <= c3x(v) + 0.4 * (1 - min(v, 10.) / 10.) + 1e-9


def test_highway_equalization_c2_blend():
  assert AC.HWY_BLEND_START == pytest.approx(35 * MPH)
  assert AC.HWY_BLEND_FULL == pytest.approx(50 * MPH)
  for v in np.linspace(50 * MPH, 45, 50):
    r = AC.smooth_accel_cap(v, "relaxed")
    assert AC.smooth_accel_cap(v, "standard") == pytest.approx(r)
    assert AC.smooth_accel_cap(v, "aggressive") == pytest.approx(r)
  # weight is C2: zero first and second derivatives at both ends
  eps = 1e-4
  for x in (AC.HWY_BLEND_START, AC.HWY_BLEND_FULL):
    w = [AC._smootherstep(AC.HWY_BLEND_START, AC.HWY_BLEND_FULL, x + k * eps) for k in (-1, 0, 1)]
    assert abs(w[2] - w[0]) / (2 * eps) < 1e-3
    assert abs(w[2] - 2 * w[1] + w[0]) / eps ** 2 < 1e-1


def test_planner_wrapper_matches_module():
  for v in np.linspace(0, 40, 41):
    assert get_max_accel(v) == pytest.approx(AC.smooth_accel_cap(v, "standard"))
    assert get_max_accel(v, True) == pytest.approx(AC.smooth_accel_cap(v, "aggressive"))
    assert get_max_accel(v, False, True) == pytest.approx(AC.smooth_accel_cap(v, "relaxed"))
    assert get_max_accel(v, True, True) == pytest.approx(AC.smooth_accel_cap(v, "relaxed"))


def test_rise_limit_only_on_increase():
  dt = 0.05
  # increase is limited to j_up*dt from max(a_prev, 0)
  assert AC.limit_accel_rise(1.0, 0.0, 0.0, dt) == pytest.approx(0.6 * dt)
  assert AC.limit_accel_rise(1.0, 0.5, 25.0, dt) == pytest.approx(0.5 + 0.45 * dt)
  # from braking, first positive step is from 0
  assert AC.limit_accel_rise(1.0, -2.0, 10.0, dt) == pytest.approx(AC.jerk_up(10.0) * dt)
  # decreases / braking pass through untouched
  for a_t, a_p in ((0.2, 0.9), (-1.0, 0.5), (-3.5, -3.5), (-0.5, -2.0), (0.0, 1.0)):
    assert AC.limit_accel_rise(a_t, a_p, 15.0, dt) == pytest.approx(a_t)
  # never above the requested target (so never above the MPC lead plan)
  for a_t in np.linspace(-3.5, 2.0, 56):
    for a_p in np.linspace(-3.5, 2.0, 12):
      assert AC.limit_accel_rise(a_t, a_p, 10.0, dt) <= a_t + 1e-12
  assert AC.jerk_up(0.0) == pytest.approx(0.6)
  assert AC.jerk_up(30.0) == pytest.approx(0.45)


def test_planner_wiring_tripwire():
  src = PLANNER.read_text()
  i = src.find("self.output_a_target = np.clip(output_a_target, ACCEL_MIN, cap)")
  j = src.find("limit_accel_rise(self.output_a_target, a_prev, v_ego, self.dt)")
  assert 0 < i < j
  assert "is_chill_aggressive(exp_mode, personality)" in src
  assert "ChillAggressiveLaunch" not in src  # no launch boost / take-up in the planner
  assert "LeadLaunch" not in src


def test_accel_sim_jerk_and_blend():
  import importlib.util
  spec = importlib.util.spec_from_file_location("accel_sim", HERE / "test3_accel_curve_sim.py")
  m = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(m)
  r = m.main()
  for p in PERS:
    for case, ba in r["drives"][p].items():
      a = ba["after"]
      assert a["peak_cmd_jerk"] <= 0.6 + 1e-6, (p, case)
      assert a["peak_accel"] <= ba["before"]["peak_accel"] + 1e-6, (p, case)
    tb = r["through_blend"][p]["after"]["summary"]
    assert tb["peak_cap_rate"] < 0.1, p
    assert tb["jerk_33_52mph"] < 0.1, p
