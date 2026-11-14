"""Curve comfort (t311u): positive accel ceiling in curves. Not road-tested."""
import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_curve_comfort import (
  A_TOTAL_MAX, CAP_RISE_JERK, LAT_ZERO, CurveComfort, long_accel_allowed, predicted_lat_accel,
)

T = np.linspace(0.0, 10.0, 33) ** 2 / 10.0  # monotone like ModelConstants.T_IDXS


def test_allowed_is_monotone_nonnegative_and_in_budget():
  lats = np.linspace(0.0, 4.0, 81)
  allowed = [long_accel_allowed(x) for x in lats]
  assert all(a >= 0.0 for a in allowed)
  assert all(b <= a + 1e-9 for a, b in zip(allowed, allowed[1:], strict=False))
  # Longitudinal never pushes total accel over the budget (lateral alone may exceed it).
  assert all(np.hypot(a, x) <= max(A_TOTAL_MAX, x) + 1e-6 for a, x in zip(allowed, lats, strict=True))
  assert long_accel_allowed(LAT_ZERO) == 0.0
  assert long_accel_allowed(0.0) == pytest.approx(A_TOTAL_MAX)


def test_prediction_sees_curve_ahead_within_lookahead():
  v = 29.0
  yaw = np.where(T >= 2.0, v / 600.0, 0.0)  # R=600 m curve 2 s ahead
  assert predicted_lat_accel(v, yaw, np.full_like(T, v), T) == pytest.approx(v * v / 600.0, rel=1e-3)
  far = np.where(T >= 5.0, v / 600.0, 0.0)  # beyond 3 s lookahead
  assert predicted_lat_accel(v, far, np.full_like(T, v), T) == 0.0
  # Ego faster than the plan: scaled to ego speed (conservative).
  assert predicted_lat_accel(v, yaw, np.full_like(T, 20.0), T) > (20.0 * 20.0 / 600.0)
  assert predicted_lat_accel(v, [], [], []) == 0.0


def test_cap_never_above_personality_cap_or_negative():
  cc = CurveComfort()
  for k in [0.0, 1 / 2000, 1 / 600, 1 / 100, 1 / 20]:
    for cap in [0.0, 0.6, 1.6, 2.0]:
      c = cc.update(cap, 20.0, k, np.zeros_like(T), np.full_like(T, 20.0), T, 0.05)
      assert 0.0 <= c <= cap


def test_straight_road_untouched_and_exit_release_is_low_jerk():
  cc = CurveComfort()
  straight = np.zeros_like(T)
  assert cc.update(1.2, 25.0, 0.0, straight, np.full_like(T, 25.0), T, 0.05) == pytest.approx(1.2)
  # Tight curve now: ceiling falls (0 at high lat).
  for _ in range(40):
    c = cc.update(1.2, 25.0, 1 / 200, straight, np.full_like(T, 25.0), T, 0.05)
  assert c == pytest.approx(0.0, abs=1e-6)
  # Exit: recovers no faster than CAP_RISE_JERK.
  prev = c
  for _ in range(100):
    c = cc.update(1.2, 25.0, 0.0, straight, np.full_like(T, 25.0), T, 0.05)
    assert c - prev <= CAP_RISE_JERK * 0.05 + 1e-9
    prev = c
  assert c == pytest.approx(1.2)


def test_closed_loop_curves_stay_in_comfort_budget():
  from openpilot.sunnypilot.selfdrive.controls.tests import test3_curve_comfort_sim as S
  res = S.main()["drives"]
  for p, d in res.items():
    for name, ba in d.items():
      b, a = ba["before"], ba["after"]
      # Never more longitudinal accel in a curve than before.
      assert a["peak_long_in_curve"] <= b["peak_long_in_curve"] + 1e-6, (p, name)
      assert a["peak_long_at_lat1"] <= b["peak_long_at_lat1"] + 1e-6, (p, name)
      # At >= 1 m/s² lateral, longitudinal stays inside the comfort budget;
      # in curves entered at >= ~1.4 m/s² lateral it stays small.
      assert a["peak_long_at_lat1"] <= long_accel_allowed(1.0) + 0.02, (p, name)
      if "R=1000" not in name:
        assert a["peak_long_at_lat1"] <= 0.35, (p, name, a["peak_long_at_lat1"])
      assert a["peak_jerk"] <= 1.6, (p, name, a["peak_jerk"])
    fw = d["freeway 65 mph, R=600 m (entry lat 1.41)"]["after"]
    assert fw["peak_total"] <= 1.65
