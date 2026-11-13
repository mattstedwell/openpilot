"""devc4-test3 Agg/Std Relaxed-like approach (LeadSlow) and PCM hold.

t304u: Relaxed apply() is a bit-identical passthrough. Agg/Std start
earlier and keep their own following distances. The Toyota PCM
smoother only runs at stopping / last ~1 mph. T_FOLLOW Agg stays
1.25. Nothing road-tested.
"""
from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

import numpy as np
import pytest
from types import SimpleNamespace

from openpilot.cereal import log
from openpilot.sunnypilot.selfdrive.car import chill_relaxed
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_lead_slow import (
  LeadSlow, is_relaxed, lead_is_slowing, slowing_a_lead_tau,
  should_use_relaxed_jerk, SLOWING_ALEAD, FINAL_STOP_JERK,
)
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_final_stop import (
  FinalStopTaper, last5_envelope, taper_comp_shift, stopping_a_lead_tau,
  FINAL_STOP_REST_DECEL, FINAL_STOP_BLEND_HI, FINAL_STOP_BLEND_LO,
  FINAL_STOP_JERK as FS_JERK, AGG_STD_STANDSTILL_M, KINEMATIC_MARGIN_M,
  TAPER_COMP_M, MPH,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  get_T_FOLLOW, get_jerk_factor,
)
from opendbc.sunnypilot.car.toyota.smooth_stop_hold import (
  smooth_pcm_accel, fade_stopping_integral, COMFORT_JERK, LAST_MPH_V,
)

P = log.LongitudinalPersonality
DT = 0.05


def _lead(dRel, vLead, aLeadK, present=True):
  return SimpleNamespace(status=present, present=present, dRel=dRel,
                         vLead=vLead, aLeadK=aLeadK, aLead=aLeadK)


def test_tfollow_aggressive_still_1_25():
  assert get_T_FOLLOW(P.aggressive, False, 20.0) == pytest.approx(1.25)
  assert get_T_FOLLOW(P.standard, False, 20.0) == pytest.approx(1.45)
  # get_jerk_factor itself is unchanged (Relaxed 1.3, Agg 0.5, Std 1.0)
  assert get_jerk_factor(P.aggressive, False) == pytest.approx(0.5)
  assert get_jerk_factor(P.standard, False) == pytest.approx(1.0)
  assert get_jerk_factor(P.relaxed, False) == pytest.approx(1.3)
  assert chill_relaxed.JERK_FACTOR == pytest.approx(1.3)


def test_relaxed_apply_is_bit_identical():
  """t304u: LeadSlow must not change Relaxed at all, including type."""
  ls = LeadSlow()
  lead = _lead(30.0, 20.0, -1.0)
  for raw in (0.0, -0.3, -1.2, -2.0, 0.4, np.float64(-0.55)):
    out = ls.apply(raw, 22.0, lead, P.relaxed, False, DT, desired_gap=25.0)
    assert out is raw or out == raw
    assert type(out) is type(raw) or (out == raw)
    # Strict identity for Python floats and numpy values we pass through:
    if type(raw) is not float or raw != 0.0:
      assert out is raw


def test_relaxed_apply_identity_many_speeds():
  ls = LeadSlow()
  for v in (1.0, 5.0, 11.2, 20.0, 31.3):
    for a in (0.0, -0.1, -0.8, -1.5, -3.2):
      raw = a
      lead = _lead(40.0, v - 1.0, -1.0)
      out = ls.apply(raw, v, lead, personality=P.relaxed, fcw=False,
                     dt=DT, desired_gap=20.0)
      assert out is raw


def test_is_relaxed_gates():
  assert is_relaxed(P.relaxed, False) is True
  assert is_relaxed(P.aggressive, False) is False
  assert is_relaxed(P.standard, False) is False
  assert is_relaxed(None, False) is True  # default = do not overlay


def test_lead_is_slowing_includes_highway_slowdown():
  # Will not halt in 15 s (70 mph at -1.0 takes ~31 s): still slowing.
  v70 = 70.0 * MPH
  assert lead_is_slowing(v70, -1.0, v70) is True
  assert lead_is_slowing(v70, -0.6, v70) is True
  assert lead_is_slowing(v70, -0.10, v70) is False  # noise, not clearly slowing
  # Stopped lead:
  assert lead_is_slowing(0.0, 0.0, 10.0) is True
  # Pulling away:
  assert lead_is_slowing(30.0, -1.0, 10.0) is False


def test_slowing_tau_persists_for_highway_slowdown():
  v70 = 70.0 * MPH
  # Official-ish tau is larger (decays faster in the prediction).
  official = 1.5
  # stopping_a_lead_tau (5ea539, Relaxed) does NOT persist a 70→55.
  assert stopping_a_lead_tau(v70, -1.0, official, v70) == official
  # Agg/Std slowing_a_lead_tau DOES persist.
  out = slowing_a_lead_tau(v70, -1.0, official, v70)
  assert out == pytest.approx(0.50)


def test_agg_starts_braking_when_lead_begins_slowing():
  """At desired gap, matched speeds, lead starts -1.0: Agg ramps at 0.30,
  does not wait at 0."""
  ls = LeadSlow()
  desired = 20.0
  lead = _lead(desired, 20.0, -1.0)
  a = 0.0
  jerks = []
  for _ in range(20):
    prev = a
    a = ls.apply(0.0, 20.0, lead, P.aggressive, False, DT, desired_gap=desired)
    jerks.append(abs(a - prev) / DT)
  # Started braking (more negative than a tiny noise).
  assert a < -0.20
  # a_lead = -1.0 raises jerk toward 0.67 (match in ~1.5 s); still
  # under the 0.90 Bae-band top, no slam.
  assert max(jerks) <= 0.90 + 1e-6
  assert a >= -0.90 - 1e-9


def test_standard_same_onset_feel():
  ls = LeadSlow()
  desired = 22.0
  lead = _lead(desired, 20.0, -1.0)
  a = 0.0
  for _ in range(10):
    a = ls.apply(0.0, 20.0, lead, P.standard, False, DT, desired_gap=desired)
  assert a < -0.10
  assert a >= -0.90 - 1e-9


def test_agg_does_not_weaken_when_short_and_closing():
  ls = LeadSlow()
  # 4 m inside desired, closing at 2 m/s, raw already -1.2
  lead = _lead(16.0, 18.0, -1.0)
  a = ls.apply(-1.2, 20.0, lead, P.aggressive, False, DT, desired_gap=20.0,
               a_prev=-1.2)
  # Never milder than raw when short and closing (jerk-limited from -1.2).
  # First frame may be -1.2 already.
  assert a <= -1.2 + 1e-9


def test_urgent_and_launch_pass_through():
  ls = LeadSlow()
  lead = _lead(3.0, 0.0, 0.0)  # under 3.5 m urgent
  assert ls.apply(-2.5, 10.0, lead, P.aggressive, False, DT, desired_gap=8.0) == -2.5
  assert ls.apply(-3.2, 10.0, lead, P.aggressive, True, DT, desired_gap=8.0) == -3.2  # FCW
  # Launch
  assert ls.apply(0.6, 0.5, lead, P.aggressive, False, DT, desired_gap=8.0) == 0.6
  # A hard raw on a comfortable gap is lowered (not passed through);
  # see test_agg_lowers_hard_mpc_peak_when_room. Tight TTC still passes.
  ls2 = LeadSlow()
  lead_ttc = _lead(8.0, 0.0, 0.0)  # 15 m/s / 8 m → TTC ≈ 0.53 s, urgent
  assert ls2.apply(-1.8, 15.0, lead_ttc, P.aggressive, False, DT, desired_gap=8.0) == -1.8


def test_should_use_relaxed_jerk_only_agg_std_slowing():
  rs = SimpleNamespace(leadOne=_lead(40.0, 30.0, -1.0))
  assert should_use_relaxed_jerk(P.aggressive, False, rs, 30.0) is True
  assert should_use_relaxed_jerk(P.standard, False, rs, 30.0) is True
  assert should_use_relaxed_jerk(P.relaxed, False, rs, 30.0) is False
  rs2 = SimpleNamespace(leadOne=_lead(40.0, 30.0, 0.0))  # not slowing
  # stopped? v=30, a=0 — not slowing, not stopping
  assert should_use_relaxed_jerk(P.aggressive, False, rs2, 30.0) is False
  # no lead
  rs3 = SimpleNamespace(leadOne=_lead(40.0, 30.0, -1.0, present=False))
  assert should_use_relaxed_jerk(P.aggressive, False, rs3, 30.0) is False


def test_final_stop_relaxed_matches_default():
  """personality=relaxed and personality=None are the 5ea539 path."""
  lead = _lead(20.0, 0.2, -0.5)
  a_none, a_rel = [], []
  fs_n, fs_r = FinalStopTaper(), FinalStopTaper()
  v = FINAL_STOP_BLEND_HI + 0.05
  raws = []
  vs = []
  # Walk 8→0
  while v > 0.0:
    raw = -0.80 if v > FINAL_STOP_BLEND_LO else -1.20
    raws.append(raw)
    vs.append(v)
    a_none.append(fs_n.apply(raw, v, lead, dt=DT, should_stop=True))
    a_rel.append(fs_r.apply(raw, v, lead, dt=DT, should_stop=True,
                            personality=P.relaxed, experimental_mode=False))
    v -= 0.25 * DT
  assert a_none == a_rel
  # And the 5ea539 rest value is still -0.08 at v=0.
  fs = FinalStopTaper()
  a0 = fs.apply(-0.4, 0.0, lead, dt=DT, should_stop=True, personality=P.relaxed)
  assert a0 == pytest.approx(-FINAL_STOP_REST_DECEL, abs=0.02) or a0 <= 0.0


def test_taper_comp_relaxed_unchanged():
  assert taper_comp_shift(1.0) == pytest.approx(TAPER_COMP_M)
  assert taper_comp_shift(1.0, personality=P.relaxed) == pytest.approx(TAPER_COMP_M)
  assert taper_comp_shift(10.0, personality=P.relaxed) == pytest.approx(0.0)
  # Agg/Std use the same 5ea539 reserved extra (24.7 ft is kinematic).
  assert taper_comp_shift(1.0, personality=P.aggressive) == pytest.approx(TAPER_COMP_M)
  assert taper_comp_shift(20.0, personality=P.aggressive) == pytest.approx(0.0)


def test_agg_kinematic_margin_is_planned_standstill():
  assert AGG_STD_STANDSTILL_M == pytest.approx(7.524, abs=0.01)
  assert KINEMATIC_MARGIN_M == pytest.approx(6.0)


def test_pcm_smooth_only_at_stop_or_last_mph():
  # Approach at 10 m/s: identity, even with a frozen-I offset.
  official = -1.00
  actuators = -0.70
  last = -0.90
  out = smooth_pcm_accel(official, actuators, last, 0.03, False, 10.0, True)
  assert out == official
  # Last mph, non-urgent: follow actuators, rate-limited at 0.30
  out = smooth_pcm_accel(-1.00, -0.08, -1.00, 0.03, True, 0.2, True)
  # step = 0.30*0.03 = 0.009, from -1.00 toward -0.08 → -0.991
  assert out == pytest.approx(-1.00 + 0.30 * 0.03)
  # Urgent: official path
  out = smooth_pcm_accel(-2.0, -2.5, -1.5, 0.03, True, 0.2, True)
  assert out == -2.0
  # Not longActive: identity
  out = smooth_pcm_accel(-1.00, -0.08, -1.00, 0.03, True, 0.2, False)
  assert out == -1.00
  # Positive (launch): identity
  out = smooth_pcm_accel(0.4, 0.4, 0.2, 0.03, False, 0.2, True)
  assert out == 0.4


def test_pcm_fade_integral():
  class _Pid:
    def __init__(self, i):
      self.i = i
  pid = _Pid(0.40)
  fade_stopping_integral(pid, 0.03)  # 0.30*0.03 = 0.009
  assert pid.i == pytest.approx(0.40 - 0.009)
  pid = _Pid(-0.005)
  fade_stopping_integral(pid, 0.03)
  assert pid.i == 0.0


def test_pcm_jerk_into_hold_at_design():
  """From a last-mph command of -0.08, ramping toward a hold, jerk is 0.30."""
  last = -0.08
  # After 1 s of 0.03 ticks, should have moved 0.30 toward a more
  # negative hold... we follow actuators which stay at -0.08, so
  # last-mph finish stays at -0.08 (no step).
  a = last
  for _ in range(10):
    a = smooth_pcm_accel(a, -0.08, a, 0.03, True, 0.0, True)
  assert a == pytest.approx(-0.08)
  # If frozen I had pushed official to -0.50 at the switch, we ease
  # from -0.50 toward -0.08 at 0.30, no step.
  a = -0.50
  jerks = []
  for _ in range(20):
    nxt = smooth_pcm_accel(-0.50, -0.08, a, 0.03, True, 0.0, True)
    jerks.append(abs(nxt - a) / 0.03)
    a = nxt
  assert max(jerks) <= COMFORT_JERK + 1e-9
  assert a > -0.50  # easing milder


def test_agg_lowers_hard_mpc_peak_when_room():
  """Comfortable already-stopped approach: raw MPC at -2.0 is lowered
  toward Relaxed's peak when kinematics allow, at 0.30 jerk. Urgent
  close-gap still passes -2.0 through."""
  ls = LeadSlow()
  # Plenty of room: 80 m, stopped lead, 25 mph, desired ~21 m
  lead = _lead(80.0, 0.0, 0.0)
  a = 0.0
  for _ in range(40):  # 2 s at 0.30 → at most -0.60 this window
    a = ls.apply(-2.0, 11.2, lead, P.aggressive, False, DT, desired_gap=21.5,
                 a_prev=a)
  # Should be braking, but not still following the raw -2.0
  assert a < -0.40
  assert a > -1.45  # capped near Relaxed peak, jerk-limited so far maybe less
  # After enough time, sits at or above (milder than) Relaxed peak
  for _ in range(80):
    a = ls.apply(-2.0, 11.2, lead, P.aggressive, False, DT, desired_gap=21.5)
  assert -1.45 <= a <= -0.50
  # Urgent close gap: passthrough
  ls2 = LeadSlow()
  lead_u = _lead(3.0, 0.0, 0.0)
  assert ls2.apply(-2.0, 11.2, lead_u, P.aggressive, False, DT, desired_gap=8.0) == -2.0
