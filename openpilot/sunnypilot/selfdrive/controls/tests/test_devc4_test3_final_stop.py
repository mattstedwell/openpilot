"""devc4-test3 chauffeur final stop and standstill hold."""
from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

import math
import numpy as np
import pytest
from types import SimpleNamespace

from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_final_stop import (
  FinalStopTaper, final_stop_envelope, last5_envelope, stopping_accel_step,
  lead_is_stopping, stopping_a_lead_tau, taper_comp_shift,
  FINAL_STOP_REST_DECEL, FINAL_STOP_BLEND_HI, FINAL_STOP_BLEND_LO,
  FINAL_STOP_JERK, FINAL_STOP_LAST5_START, STOPPING_DECEL_RATE_STANDSTILL,
  TAPER_COMP_M, MPH,
)

DT = 0.05


def test_envelope_at_standstill_is_near_zero():
  e0 = final_stop_envelope(0.0)
  assert e0 == pytest.approx(-FINAL_STOP_REST_DECEL)
  assert -0.10 <= e0 <= -0.05  # Matt's chauffeur window


def test_envelope_flat_in_regen_blend_band():
  # No commanded change 5-8 mph: envelope is constant.
  vals = [final_stop_envelope(v) for v in np.linspace(FINAL_STOP_BLEND_LO, FINAL_STOP_BLEND_HI, 9)]
  assert max(vals) - min(vals) < 1e-9
  assert vals[0] == pytest.approx(-FINAL_STOP_LAST5_START)


def test_envelope_tapers_last_5_mph():
  a5 = last5_envelope(FINAL_STOP_BLEND_LO, -0.50)
  a0 = last5_envelope(0.0, -0.50)
  assert a5 == pytest.approx(-0.50)
  assert a0 == pytest.approx(-FINAL_STOP_REST_DECEL)
  # Monotone (becomes less negative as v falls) for a start of -0.50.
  prev = a5
  for v in np.linspace(FINAL_STOP_BLEND_LO, 0.0, 20):
    a = last5_envelope(v, -0.50)
    assert a >= prev - 1e-9
    prev = a


def test_blend_band_holds_command_constant():
  fs = FinalStopTaper()
  lead = SimpleNamespace(status=True, present=True, dRel=20.0, vLead=0.2, aLeadK=-0.5)
  # Arrive at just under 8 mph with -0.70, then drop through the band.
  a = fs.apply(-0.70, FINAL_STOP_BLEND_HI + 0.05, lead, dt=DT, should_stop=False)
  assert a == pytest.approx(-0.70)  # above 8: pass through
  held = None
  jerks = []
  prev_a, prev_v = a, FINAL_STOP_BLEND_HI + 0.05
  v = FINAL_STOP_BLEND_HI - 0.01
  while v > FINAL_STOP_BLEND_LO + 0.01:
    a = fs.apply(-1.20, v, lead, dt=DT, should_stop=False)  # raw wants to change
    if held is None:
      held = a
    else:
      assert a == pytest.approx(held, abs=1e-9)
    jerks.append(abs(a - prev_a) / DT)
    prev_a, prev_v = a, v
    v -= 0.20 * DT  # slow crawl through the band
  assert max(jerks) < 1e-6  # no commanded change


def test_last_5_mph_jerk_at_or_under_design():
  fs = FinalStopTaper()
  lead = SimpleNamespace(status=True, present=True, dRel=15.0, vLead=0.0, aLeadK=0.0)
  # Freeze a start of -0.80 through the band, then taper.
  fs.apply(-0.80, FINAL_STOP_BLEND_HI - 0.05, lead, dt=DT, should_stop=True)
  v = FINAL_STOP_BLEND_LO - 0.01
  prev = fs.prev
  peak_j = 0.0
  while v > 0.0:
    a = fs.apply(-1.20, v, lead, dt=DT, should_stop=True)
    j = abs(a - prev) / DT
    peak_j = max(peak_j, j)
    # Finished near rest, never positive.
    assert a <= 0.0
    prev = a
    # kinematics under this accel
    v = max(0.0, v + a * DT)
  assert peak_j <= FINAL_STOP_JERK + 1e-6
  assert prev == pytest.approx(-FINAL_STOP_REST_DECEL, abs=0.02)


def test_launch_positive_passes_through():
  fs = FinalStopTaper()
  lead = SimpleNamespace(status=True, present=True, dRel=8.0, vLead=0.05, aLeadK=0.3)
  # A stopped-looking lead with a positive launch preview must not be
  # clamped to 0.
  a = fs.apply(0.10, 0.0, lead, dt=DT, should_stop=False)
  assert a == pytest.approx(0.10)
  a = fs.apply(0.30, 0.1, lead, dt=DT, should_stop=False)
  assert a == pytest.approx(0.30)


def test_urgent_fcw_and_close_gap_pass_through():
  fs = FinalStopTaper()
  lead = SimpleNamespace(status=True, present=True, dRel=20.0, vLead=2.0, aLeadK=-1.0)
  assert fs.apply(-2.5, 5.0, lead, fcw=True, dt=DT) == pytest.approx(-2.5)
  lead.dRel = 2.0  # under 3.5 m fallback (and under any personality floor)
  assert fs.apply(-2.0, 5.0, lead, fcw=False, dt=DT) == pytest.approx(-2.0)
  # Short TTC: 8 m gap, closing at 5 m/s → TTC 1.6 s < 2
  lead.dRel = 8.0
  lead.vLead = 5.0
  assert fs.apply(-2.2, 10.0, lead, fcw=False, dt=DT) == pytest.approx(-2.2)


def test_stopping_accel_step_rolling_follows_envelope_not_stopaccel():
  # Official is -1.0 * dt toward stopAccel (-2.0) while rolling.
  # We must stay near the chauffeur envelope and not dive to -2.
  a = -0.10
  v = 0.20  # just above vEgoStopping, still rolling
  for _ in range(20):
    a = stopping_accel_step(a, standstill=False, dt=0.01, v_ego=v, stop_accel=-2.0)
    v = max(0.0, v + a * 0.01)
  # After 0.2 s rolling, still near rest decel, far from -2.0
  assert a == pytest.approx(final_stop_envelope(v), abs=0.05)
  assert a > -0.5


def test_stopping_accel_step_standstill_ramps_slowly():
  a = -FINAL_STOP_REST_DECEL
  dt = 0.01
  a1 = stopping_accel_step(a, standstill=True, dt=dt, v_ego=0.0, stop_accel=-2.0)
  assert a1 == pytest.approx(a - STOPPING_DECEL_RATE_STANDSTILL * dt)
  # Official 1.0 rate would have moved 0.01; we move 0.003.
  assert abs(a1 - a) == pytest.approx(0.003, abs=1e-9)
  # After many frames we reach stopAccel and the caller clips.
  for _ in range(800):
    a = stopping_accel_step(a, standstill=True, dt=dt, v_ego=0.0, stop_accel=-2.0)
  assert a <= -2.0 + 1e-9  # unclipped would pass -2; caller clips
  # Rate is 0.30: 1.92 / 0.30 ≈ 6.4 s from -0.08 to -2.0
  a = -FINAL_STOP_REST_DECEL
  t = 0.0
  while a > -2.0 and t < 20.0:
    a = stopping_accel_step(a, standstill=True, dt=dt, v_ego=0.0, stop_accel=-2.0)
    t += dt
  assert 6.0 < t < 7.0


def test_stopping_accel_step_does_not_weaken_urgent():
  a = stopping_accel_step(-1.8, standstill=False, dt=0.01, v_ego=0.4, stop_accel=-2.0)
  assert a == pytest.approx(-1.8)


def test_stopping_lead_helpers():
  assert lead_is_stopping(0.2, 0.0) is True
  assert lead_is_stopping(10.0, -0.2) is False  # won't halt in 15 s
  assert lead_is_stopping(8.0, -1.0) is True
  assert stopping_a_lead_tau(5.0, -1.0, 1.5, 8.0) == pytest.approx(0.50)
  assert stopping_a_lead_tau(5.0, 0.0, 1.5, 15.0) == pytest.approx(1.5)
  assert taper_comp_shift(1.0) == pytest.approx(TAPER_COMP_M)
  assert taper_comp_shift(10.0) == pytest.approx(0.0)


def test_no_creep():
  fs = FinalStopTaper()
  lead = SimpleNamespace(status=True, present=True, dRel=8.0, vLead=0.0, aLeadK=0.0)
  a = 0.0
  v = FINAL_STOP_BLEND_LO
  for _ in range(80):
    a = fs.apply(-0.05, v, lead, dt=DT, should_stop=True)
    assert a <= 0.0
    v = max(0.0, v + a * DT)
