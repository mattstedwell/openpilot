from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import custom
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, long_control_state_trans


class TestLongControlStateTransition(OpenpilotTestCase):

  def test_stay_stopped(self):
    CP_SP = custom.CarParamsSP.new_message()
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

  def test_engage(self):
    CP_SP = custom.CarParamsSP.new_message()
    active = True
    current_state = LongCtrlState.off
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid

  def test_auto_resume_sng_not_blocked_by_cruise_standstill(self):
    """Cars with autoResumeSng (Toyota TSS2 NO_STOP_TIMER) must leave Stopping
    when the planner wants to move, even if the PCM still reports cruise
    standstill. Otherwise we never send the positive accel that releases the
    PCM, and a gas tap is required."""
    CP_SP = custom.CarParamsSP.new_message()
    current_state = LongCtrlState.stopping
    # without the flag, cruise standstill still blocks (stock / non-SnG cars)
    next_state = long_control_state_trans(CP_SP, True, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True,
                             auto_resume_sng=False)
    assert next_state == LongCtrlState.stopping
    # with the flag, same inputs go to PID
    next_state = long_control_state_trans(CP_SP, True, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True,
                             auto_resume_sng=True)
    assert next_state == LongCtrlState.pid
    # should_stop or brake still hold the car
    next_state = long_control_state_trans(CP_SP, True, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=True,
                             auto_resume_sng=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, True, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=True,
                             auto_resume_sng=True)
    assert next_state == LongCtrlState.stopping
