"""devc4-test3 launch take-up (test2 LaunchStart + predictive go / preview).

Host-safe: numpy only. Does not cover test2's +8 ft stop, LeadDecel,
30%-PWR caps, cut-in, catch-up, or final-stop. Caps used here are
test3's C3X numbers. Nothing road-tested.
"""
import math

import pytest

from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

from openpilot.cereal import log
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_launch import (
  LaunchStart, LeadLaunch, launch_go, launch_hard, _lead_departing,
  LAUNCH_START_STEP, LAUNCH_JERK_UP, LAUNCH_GO_CONFIRM, LAUNCH_GO_V_EGO_MAX,
  LAUNCH_GO_MIN_GAP, LAUNCH_PREVIEW, LAUNCH_GO_A_LEAD, DT, MPH,
  LAUNCH_GO_V_EGO_HOLD,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_lead_stop_extra

P = log.LongitudinalPersonality
FT = 0.3048


class _Lead:
  def __init__(self, d=8.0, v=0.0, a=0.0, status=True):
    self.status = status
    self.present = status
    self.dRel = d
    self.vLead = v
    self.aLeadK = a


def _step_launch(ls, go, a_chosen=-0.3, cap=1.28, jerk=LAUNCH_JERK_UP, n=1, **kw):
  a = a_chosen
  for _ in range(n):
    a = ls.update(go, a_chosen, cap, jerk, DT, **kw)
  return a


# C3X / this branch's caps at a stop. Aggressive includes the +0.4 boost.
CAP_RELAXED = get_max_accel(0.0, chill_relaxed_mode=True)   # 1.28
CAP_STANDARD = get_max_accel(0.0)                           # 1.6
CAP_AGGRESSIVE = get_max_accel(0.0, True)                   # 2.0


# --- launch_go ---

def test_launch_go_stopped_lead_false():
  for d in (6.0, 7.524, 9.048, 20.0):
    assert launch_go(True, d, 0.0, 0.0, 25 * MPH) is False
    assert launch_go(True, d, 0.05, 0.0, 25 * MPH) is False


def test_launch_go_lead_moving_away():
  assert launch_go(True, 8.0, 0.20, 0.0, 25 * MPH) is True
  assert launch_go(True, 7.524, 0.50, 0.0, 25 * MPH) is True
  # already rolling, lead pulling away, under ~10 mph
  assert launch_go(True, 10.0, 2.34, 2.24, 25 * MPH) is True


def test_launch_go_closing_or_matched_false():
  assert launch_go(True, 8.0, 0.20, 0.20, 25 * MPH) is False  # not pulling away
  assert launch_go(True, 8.0, 0.10, 0.20, 25 * MPH) is False  # closing


def test_launch_go_above_10mph_false():
  assert launch_go(True, 15.0, 6.0, 5.0, 25 * MPH) is False  # ~11 mph
  assert launch_go(True, 15.0, 6.0, LAUNCH_GO_V_EGO_MAX + 0.1, 25 * MPH) is False


def test_launch_go_close_gap_and_fcw():
  assert launch_go(True, LAUNCH_GO_MIN_GAP - 0.1, 1.0, 0.0, 25 * MPH) is False
  assert launch_go(True, 8.0, 1.0, 0.0, 25 * MPH, fcw=True) is False


def test_launch_go_nolead():
  assert launch_go(False, float("nan"), float("nan"), 0.0, 25 * MPH) is True
  assert launch_go(False, float("nan"), float("nan"), 0.0, 0.2) is False


def test_launch_go_predictive_accel():
  # Clearly accelerating lead is go even with v_lead still under 0.10.
  assert launch_go(True, 7.524, 0.01, 0.0, 25 * MPH, a_lead=1.0) is True
  assert launch_go(True, 7.524, 0.01, 0.0, 25 * MPH, a_lead=0.5) is True
  assert launch_go(True, 7.524, 0.0, 0.0, 25 * MPH, a_lead=1.0) is True
  # Not accelerating enough, or a stopped lead (a≈0).
  assert launch_go(True, 7.524, 0.01, 0.0, 25 * MPH, a_lead=0.10) is False
  assert launch_go(True, 7.524, 0.0, 0.0, 25 * MPH, a_lead=0.0) is False
  assert launch_go(True, 7.524, 0.0, 0.0, 25 * MPH) is False  # a_lead missing
  # Closing on us: not go.
  assert launch_go(True, 7.524, 0.01, 0.20, 25 * MPH, a_lead=1.0) is False
  assert _lead_departing(0.01, 0.0, 1.0) is True
  assert _lead_departing(0.0, 0.0, 0.0) is False
  # a_lead == threshold but v_pred = 0 + 0.25*0.20 = 0.05, not above 0.10
  assert _lead_departing(0.0, 0.0, LAUNCH_GO_A_LEAD) is False


# --- LaunchStart ---

def test_first_step_after_confirm():
  ls = LaunchStart()
  a = _step_launch(ls, True, n=1)
  assert a == pytest.approx(LAUNCH_PREVIEW)  # brake-release preview, not the 0.30 step
  assert ls.active is False
  assert ls.engaging is True
  assert math.isclose(LAUNCH_PREVIEW, 0.10)
  a = _step_launch(ls, True, n=1)
  assert a == pytest.approx(LAUNCH_START_STEP)
  assert ls.active is True
  assert math.isclose(LAUNCH_START_STEP, 0.30)


def test_single_frame_go_does_not_step():
  ls = LaunchStart()
  a = _step_launch(ls, True, n=1)
  assert a == pytest.approx(LAUNCH_PREVIEW)
  assert ls.active is False
  # go drops: preview is abandoned, back to the chosen hold, no 0.30 step
  a = ls.update(False, -0.3, CAP_RELAXED, LAUNCH_JERK_UP, DT)
  assert a == pytest.approx(-0.3)
  assert ls.active is False
  assert ls.engaging is False
  assert ls.go_count == 0


def test_ramp_at_launch_jerk_then_cap():
  ls = LaunchStart()
  a = _step_launch(ls, True, n=LAUNCH_GO_CONFIRM, cap=CAP_RELAXED)
  assert a == pytest.approx(0.30)
  # next frame climbs at 0.4 m/s³ * 0.05 s = 0.02
  a = ls.update(True, -0.3, CAP_RELAXED, LAUNCH_JERK_UP, DT)
  assert a == pytest.approx(0.30 + LAUNCH_JERK_UP * DT)
  for _ in range(200):
    a = ls.update(True, -0.3, CAP_RELAXED, LAUNCH_JERK_UP, DT)
  assert a == pytest.approx(CAP_RELAXED)
  # time from step to cap: (1.28-0.30)/0.4 = 2.45 s


def test_false_start_releases_immediately():
  ls = LaunchStart()
  _step_launch(ls, True, n=LAUNCH_GO_CONFIRM)
  a = ls.update(True, -0.3, CAP_RELAXED, LAUNCH_JERK_UP, DT, hard=True)
  assert a == pytest.approx(-0.3)
  assert ls.active is False


def test_urgent_plan_never_lifted():
  ls = LaunchStart()
  a = _step_launch(ls, True, a_chosen=-1.8, n=5)
  assert a == pytest.approx(-1.8)
  assert ls.active is False


def test_does_not_step_down_if_already_above():
  ls = LaunchStart()
  a = _step_launch(ls, True, a_chosen=0.55, n=LAUNCH_GO_CONFIRM)
  assert a == pytest.approx(0.55)


@pytest.mark.parametrize("cap", [CAP_RELAXED, CAP_STANDARD, CAP_AGGRESSIVE])
def test_c3x_personality_caps(cap):
  """Envelope never exceeds this branch's C3X caps (Rel 1.28, Std 1.6, Agg 2.0)."""
  ls = LaunchStart()
  a = _step_launch(ls, True, n=LAUNCH_GO_CONFIRM, cap=cap)
  assert a == pytest.approx(min(LAUNCH_START_STEP, cap))
  for _ in range(400):
    a = ls.update(True, -0.3, cap, LAUNCH_JERK_UP, DT)
  assert a == pytest.approx(cap)
  assert a <= cap + 1e-9


def test_aggressive_boost_is_the_cap_not_a_noop():
  """Aggressive at a stop is 2.0 (C3X curve + 0.4); Standard is 1.6."""
  assert CAP_AGGRESSIVE == pytest.approx(2.0)
  assert CAP_STANDARD == pytest.approx(1.6)
  assert CAP_RELAXED == pytest.approx(1.28)
  ls_a = LaunchStart()
  ls_s = LaunchStart()
  for _ in range(400):
    a_a = ls_a.update(True, -0.3, CAP_AGGRESSIVE, LAUNCH_JERK_UP, DT)
    a_s = ls_s.update(True, -0.3, CAP_STANDARD, LAUNCH_JERK_UP, DT)
  assert a_a == pytest.approx(2.0)
  assert a_s == pytest.approx(1.6)
  assert a_a > a_s


def test_latency_to_positive_request():
  """First positive request is the preview at 0.05 s; the 0.30 step is
  at confirm * DT = 0.10 s. Both are well under the old ~1.7-1.9 s t_move."""
  ls = LaunchStart()
  t = 0.0
  a = -0.4
  t_pos = t_step = None
  for _ in range(40):
    a = ls.update(True, -0.4, CAP_RELAXED, LAUNCH_JERK_UP, DT)
    t += DT
    if a > 0.0 and t_pos is None:
      t_pos = t
    if a >= LAUNCH_START_STEP - 1e-9 and t_step is None:
      t_step = t
  assert t_pos == pytest.approx(DT)  # preview, first go frame
  assert t_pos <= 0.05
  assert t_step == pytest.approx(LAUNCH_GO_CONFIRM * DT)
  assert t_step <= 0.10


# --- LeadLaunch (wrapper, no LeadDecel) ---

def test_no_creep_on_stopped_lead():
  ls = LeadLaunch()
  lead = _Lead(d=7.524, v=0.0, a=0.0)
  a, ss = 0.0, True
  for _ in range(40):
    a, ss = ls.apply(-0.5, True, 0.0, 25 * MPH, lead, False, DT, cap=CAP_RELAXED)
  assert a == pytest.approx(-0.5)
  assert ss is True
  assert ls.launch.active is False
  assert ls.launch.engaging is False


def test_follow_launches_then_should_stop_clears():
  ls = LeadLaunch()
  lead = _Lead(d=7.524, v=0.5, a=0.5)
  a, ss = -0.5, True
  t_pos = t_ss = t_step = None
  t = 0.0
  for _ in range(20):
    a, ss = ls.apply(-0.5, True, 0.0, 25 * MPH, lead, False, DT, cap=CAP_RELAXED)
    t += DT
    if a > 0.0 and t_pos is None:
      t_pos = t
    if (not ss) and t_ss is None:
      t_ss = t
    if a >= LAUNCH_START_STEP - 1e-9 and t_step is None:
      t_step = t
  assert t_pos == pytest.approx(0.05)  # preview
  assert t_ss == pytest.approx(0.05)   # should_stop clears on preview (a >= 0.1)
  assert t_step == pytest.approx(0.10)
  assert a >= LAUNCH_START_STEP - 1e-9
  assert ss is False


def test_false_start_then_stopped_lead_releases():
  ls = LeadLaunch()
  lead = _Lead(d=7.524, v=0.5, a=0.4)
  for _ in range(6):
    a, ss = ls.apply(-0.3, True, 0.0, 25 * MPH, lead, False, DT, cap=CAP_RELAXED)
  assert a > 0.0
  # lead stops
  lead.vLead = 0.0
  lead.aLeadK = -1.0
  a, ss = ls.apply(-0.8, True, 0.2, 25 * MPH, lead, False, DT, cap=CAP_RELAXED)
  assert a == pytest.approx(-0.8)  # hard release
  assert ls.launch.active is False


def test_launch_hard_helpers():
  assert launch_hard(True, 8.0, 0.0, fcw=False, a_chosen=0.0) is True
  assert launch_hard(True, 3.0, 1.0, fcw=False, a_chosen=0.0) is True
  assert launch_hard(True, 8.0, 1.0, fcw=True, a_chosen=0.0) is True
  assert launch_hard(True, 8.0, 1.0, fcw=False, a_chosen=-1.2) is True
  assert launch_hard(True, 8.0, 1.0, fcw=False, a_chosen=0.2) is False
  # A just-departing lead (v still < 0.05, a clearly positive) is not a hard stop.
  assert launch_hard(True, 8.0, 0.01, fcw=False, a_chosen=-0.5, a_lead=1.0) is False
  assert launch_hard(True, 8.0, 0.0, fcw=False, a_chosen=-0.5, a_lead=0.0) is True


def test_preview_does_not_creep_on_stopped_lead():
  """Predictive go + preview must not roll toward a stopped lead."""
  ls = LeadLaunch()
  lead = _Lead(d=9.048, v=0.0, a=0.0)
  a, ss = 0.0, True
  for _ in range(40):
    a, ss = ls.apply(-0.5, True, 0.0, 25 * MPH, lead, False, DT, cap=CAP_RELAXED)
  assert a == pytest.approx(-0.5)
  assert ss is True
  assert ls.launch.active is False
  assert ls.launch.engaging is False


def test_early_go_on_half_g_pullaway_from_rest():
  """0.5 m/s² lead at v≈0 is go on the first frame; preview then 0.30 step."""
  ls = LeadLaunch()
  lead = _Lead(d=7.524, v=0.01, a=0.5)
  a, ss = ls.apply(-0.5, True, 0.0, 25 * MPH, lead, False, DT, cap=CAP_RELAXED)
  assert a == pytest.approx(LAUNCH_PREVIEW)
  assert ss is False
  assert ls.launch.active is False
  a, ss = ls.apply(-0.5, True, 0.0, 25 * MPH, lead, False, DT, cap=CAP_RELAXED)
  assert a == pytest.approx(LAUNCH_START_STEP)
  assert ss is False
  assert ls.launch.active is True


def test_stop_extras_and_gaps_unchanged():
  """This port must not change test3's +5/+5/+10 ft extras."""
  assert get_lead_stop_extra(P.relaxed, False) == pytest.approx(10 * FT)
  assert get_lead_stop_extra(P.standard, False) == pytest.approx(5 * FT)
  assert get_lead_stop_extra(P.aggressive, False) == pytest.approx(5 * FT)
  assert get_lead_stop_extra(P.aggressive, True) == 0.0


def test_already_active_does_not_stay_on_at_city_speeds():
  """The bug Matt felt: already_active ignored v_ego and stayed in
  take-up through 25-30 mph, ramping to cap and closing the gap."""
  # New engagement above 10 mph is already False (existing test).
  # already_active at 15 / 25 / 30 mph must also be False, even when
  # the lead is matching us (the old CATCH hysteresis).
  for v in (15 * MPH, 25 * MPH, 30 * MPH, 12.0):
    assert launch_go(True, 20.0, v, v, 25 * MPH, already_active=True) is False
  # A little past 10 mph may still hold (hysteresis, 11.2 mph ceiling).
  v_hold = LAUNCH_GO_V_EGO_MAX + 0.2  # 4.7 m/s, under 5.0 hold
  assert v_hold < LAUNCH_GO_V_EGO_HOLD
  assert launch_go(True, 20.0, v_hold + 0.1, v_hold, 25 * MPH, already_active=True) is True
  # At the ceiling, no.
  assert launch_go(True, 20.0, LAUNCH_GO_V_EGO_HOLD + 0.2,
                   LAUNCH_GO_V_EGO_HOLD + 0.1, 25 * MPH, already_active=True) is False


def test_desired_gap_gate_hands_back_to_mpc():
  """Take-up must not keep ramping to cap once the gap is below desired."""
  ls = LeadLaunch()
  # Establish an active launch at a comfortable gap.
  lead = _Lead(d=10.0, v=1.0, a=0.5)
  ls.apply(-0.5, True, 0.0, 25 * MPH, lead, False, DT, cap=1.6, desired_gap=8.0)
  ls.apply(-0.5, True, 0.0, 25 * MPH, lead, False, DT, cap=1.6, desired_gap=8.0)
  assert ls.launch.active is True
  # Now the gap is below desired: go is forced off, we must not
  # command the cap. Soft-exit / handoff toward the chosen plan.
  lead = _Lead(d=8.0, v=2.0, a=0.5)
  a, ss = ls.apply(-0.1, False, 1.0, 25 * MPH, lead, False, DT, cap=1.6,
                   desired_gap=12.0)
  # Not ramping toward 1.6.
  assert a <= 0.30 + 1e-6 or a <= 0.0  # handed toward chosen (-0.1) or at most the step
  # After a few frames we should be at or near the chosen plan, not the cap.
  for _ in range(30):
    a, ss = ls.apply(-0.1, False, 2.0, 25 * MPH, lead, False, DT, cap=1.6,
                     desired_gap=12.0)
  assert a == pytest.approx(-0.1, abs=0.05)
  assert a < 0.5  # definitely not the 1.6 cap


def test_floor_is_hard_release():
  ls = LeadLaunch()
  lead = _Lead(d=10.0, v=1.0, a=0.5)
  ls.apply(-0.5, True, 0.0, 25 * MPH, lead, False, DT, cap=1.6, min_gap_floor=7.524)
  ls.apply(-0.5, True, 0.0, 25 * MPH, lead, False, DT, cap=1.6, min_gap_floor=7.524)
  assert ls.launch.active is True
  lead = _Lead(d=5.0, v=1.0, a=0.5)  # inside 24.69 ft floor
  a, ss = ls.apply(-0.8, False, 1.0, 25 * MPH, lead, False, DT, cap=1.6,
                   min_gap_floor=7.524)
  assert a == pytest.approx(-0.8)  # hard release: raw chosen plan
  assert ls.launch.active is False
