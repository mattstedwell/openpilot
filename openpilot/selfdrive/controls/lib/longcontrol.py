import numpy as np
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState

# sunnypilot-lite smooth final stop (Matt: "make the braking at full stops near
# zero more comfortable ... too jerky"). Stock Stopping ramps the command toward
# stopAccel (-2.0) at 1.0 m/s^2/s from the moment should_stop goes true
# (v < 0.3 m/s), while the car is still rolling, so the last ~0.7 s of every
# stop grabs from ~-0.2 to ~-0.7 m/s^2 right at the stop. While still rolling we
# now follow a speed-tapered finish instead: -(STOP_ROLL_DECEL + STOP_ROLL_K * v),
# about -0.45 m/s^2 at 0.3 m/s easing toward -0.15 m/s^2 at the stop (sim: ~-0.25
# at the stop instant vs stock ~-0.65), never weaker than the plan, and only build
# the standstill hold toward stopAccel once the car is at rest (stock rate).
# Harder plan braking passes straight through. The few cm of extra roll are
# covered by long_mpc STOP_TAPER_COMP (same real-approach latch) and
# lite_e2e_stop EXTRA, so the stop point does not move toward the lead / line.
# Only for a real decelerating approach: the car must have been above
# STOP_SOFT_V_MIN since its last standstill and already be braking
# (aEgo < STOP_SOFT_AEGO_MAX) when Stopping begins. Short creeps / roll-stop
# cycles keep the stock firm ramp (sim showed the soft finish let creep cycles
# end closer to the lead).
STOP_SMOOTH = True
STOP_SOFT_AEGO_MAX = -0.10  # m/s^2
STOP_SOFT_V_MIN = 2.0       # m/s (~4.5 mph)
STOP_ROLL_DECEL = 0.15     # m/s^2: brake at the moment of the stop
STOP_ROLL_K = 1.0          # 1/s: extra brake per m/s of speed while rolling
STOP_ROLL_JERK_IN = 1.0    # m/s^3: build rate (stock Stopping rate)
STOP_ROLL_JERK = 0.6       # m/s^3: ease rate toward the gentle finish
STOP_ROLL_URGENT = 0.30    # m/s^2: plan this much harder than the command -> follow at once
STOP_REST_V = 0.02         # m/s: at rest (or CS.standstill)


def long_control_state_trans(CP_SP, active, long_control_state,
                             should_stop, brake_pressed, cruise_standstill):
  # Gas Interceptor
  cruise_standstill = cruise_standstill and not CP_SP.enableGasInterceptor

  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.pid:
      if should_stop:
        long_control_state = LongCtrlState.stopping

  return long_control_state

class LongControl:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController(0.0, (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.stop_soft = False
    self.v_peak = 0.0  # max speed since the last standstill

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    prev_state = self.long_control_state
    self.long_control_state = long_control_state_trans(self.CP_SP, active, self.long_control_state,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill)
    self.v_peak = 0.0 if CS.standstill else max(self.v_peak, float(CS.vEgo))
    if self.long_control_state != LongCtrlState.stopping:
      self.stop_soft = False
    elif prev_state != LongCtrlState.stopping:
      self.stop_soft = bool(STOP_SMOOTH and float(CS.aEgo) < STOP_SOFT_AEGO_MAX and self.v_peak >= STOP_SOFT_V_MIN)
    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping and self.stop_soft and \
         not (CS.standstill or CS.vEgo < STOP_REST_V):
      # still rolling into the stop: gentle, smooth finish (see STOP_SMOOTH)
      output_accel = min(self.last_output_accel, 0.0)
      tgt = min(float(a_target), -(STOP_ROLL_DECEL + STOP_ROLL_K * max(float(CS.vEgo), 0.0)))
      if float(a_target) < output_accel - STOP_ROLL_URGENT:
        output_accel = tgt
      elif tgt < output_accel:
        output_accel = max(tgt, output_accel - STOP_ROLL_JERK_IN * DT_CTRL)
      else:
        output_accel = min(tgt, output_accel + STOP_ROLL_JERK * DT_CTRL)
      self.reset()

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        # TODO: can we just go straight to stopAccel?
        output_accel -= 1.0 * DT_CTRL  # m/s^2/s while trying to stop
      self.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo,
                                     feedforward=a_target)

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
