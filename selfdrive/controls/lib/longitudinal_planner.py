#!/usr/bin/env python3
import math
import numpy as np

import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource, CRUISE_MAX_ACCEL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.sunnypilot.selfdrive.car.chill_aggressive import ChillAggressiveLaunch

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# Hold back this far from the e2e model's planned stop for lights/signs.
E2E_INTERSECTION_STOP_EXTRA = 15 * 0.3048  # 15 feet
E2E_PLANNED_STOP_V = 0.5  # m/s
E2E_LEAD_STOP_MARGIN = 5.0  # m

# After a confirmed green, use chill/MPC accel instead of the model's timid launch.
E2E_GREEN_PATH_X = 25 * 0.3048  # 25 feet
E2E_LAUNCH_BRAKE_YIELD = -0.3
# Require a real stop-then-go (green) before launching from a light.
E2E_GREEN_ARM_S = 0.6
E2E_GREEN_CONFIRM_S = 0.1
E2E_GREEN_STOP_X = 50.0
E2E_GREEN_FUTURE_STOP_X = 2.0

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego, chill_aggressive_launch=False):
  stock = np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)
  return stock + (ChillAggressiveLaunch.accel_extra(v_ego) if chill_aggressive_launch else 0.)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False
    self.chill_aggressive_launch = ChillAggressiveLaunch()
    self._from_standstill = False
    self._armed_for_green = False
    self._red_wait_elapsed = 0.0
    self._green_confirm_elapsed = 0.0
    self._green_confirmed = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  @staticmethod
  def _adjust_e2e_intersection_stop(sm, v_ego, a_e2e, should_stop_e2e):
    # Stop 15 ft earlier than the e2e plan at lights and stop signs, and hold
    # there so the car does not creep up to the original stop line.
    model = sm['modelV2']
    extra = E2E_INTERSECTION_STOP_EXTRA
    if len(model.position.x) < 2 or len(model.velocity.x) < 2:
      return a_e2e, should_stop_e2e

    x = np.asarray(model.position.x, dtype=float)
    v = np.asarray(model.velocity.x, dtype=float)
    n = min(len(x), len(v))
    x = x[:n]
    v = v[:n]

    stop_idxs = np.flatnonzero(v < E2E_PLANNED_STOP_V)
    if stop_idxs.size == 0:
      return a_e2e, should_stop_e2e

    i_stop = int(stop_idxs[0])
    x_stop = float(x[i_stop])

    lead = sm['radarState'].leadOne
    if lead.status and float(lead.dRel) < x_stop + E2E_LEAD_STOP_MARGIN:
      return a_e2e, should_stop_e2e

    target_x = x_stop - extra
    v_ego = max(float(v_ego), 0.0)

    if target_x <= 0.8:
      if v_ego > 0.15:
        a_hold = -(v_ego * v_ego) / (2.0 * max(target_x, 0.4))
      else:
        a_hold = min(float(a_e2e), 0.0)
      return min(float(a_e2e), a_hold), True

    x_seg = np.maximum.accumulate(x[:i_stop + 1])
    v_seg = v[:i_stop + 1]
    if len(model.acceleration.x) >= n:
      a_seg = np.asarray(model.acceleration.x, dtype=float)[:i_stop + 1]
      a_shifted = float(np.interp(extra, x_seg, a_seg))
    else:
      a_shifted = float(a_e2e)

    v_shifted = float(np.interp(extra, x_seg, v_seg))
    return min(float(a_e2e), a_shifted), bool(should_stop_e2e) or (v_shifted < 0.3)

  def _reset_green_wait(self):
    self._armed_for_green = False
    self._red_wait_elapsed = 0.0
    self._green_confirm_elapsed = 0.0
    self._green_confirmed = False

  @staticmethod
  def _nearby_intersection_stop(sm):
    # True when the model still plans to stop for a light or sign, not a lead.
    model = sm['modelV2']
    if len(model.position.x) < 2 or len(model.velocity.x) < 2:
      return False
    x = np.asarray(model.position.x, dtype=float)
    v = np.asarray(model.velocity.x, dtype=float)
    n = min(len(x), len(v))
    x = x[:n]
    v = v[:n]
    future_stop = (x > E2E_GREEN_FUTURE_STOP_X) & (v < E2E_PLANNED_STOP_V)
    if not np.any(future_stop):
      return False
    x_stop = float(x[int(np.flatnonzero(future_stop)[0])])
    if x_stop >= E2E_GREEN_STOP_X:
      return False
    lead = sm['radarState'].leadOne
    if lead.status and float(lead.dRel) < x_stop + E2E_LEAD_STOP_MARGIN:
      return False
    return True

  def _update_e2e_chill_launch(self, sm, v_ego, v_cruise, output_a_target_e2e, output_should_stop_mpc, reset_state):
    # Launch from a light only after the model has been holding a stop and
    # then clearly releases it (the closest we have to a green-light signal).
    # An open-looking path alone is not enough, because the model can see
    # through an intersection while the light is still red.
    if reset_state:
      self._from_standstill = False
      self._reset_green_wait()
      return False

    if sm['carState'].standstill:
      if not self._from_standstill:
        self._reset_green_wait()
      self._from_standstill = True
    elif v_cruise > 0 and v_ego >= max(v_cruise - 1.0, 0.0):
      self._from_standstill = False
      self._reset_green_wait()

    model_x = sm['modelV2'].position.x
    path_open = len(model_x) > 0 and float(model_x[-1]) > E2E_GREEN_PATH_X
    should_stop_e2e = bool(sm['modelV2'].action.shouldStop)
    nearby_stop = self._nearby_intersection_stop(sm)
    standstill = bool(sm['carState'].standstill)
    waiting_on_light = should_stop_e2e or nearby_stop

    if standstill and waiting_on_light:
      self._red_wait_elapsed += self.dt
      if self._red_wait_elapsed >= E2E_GREEN_ARM_S:
        self._armed_for_green = True
      self._green_confirm_elapsed = 0.0
    elif standstill and self._armed_for_green and path_open and (not waiting_on_light) and (not output_should_stop_mpc):
      self._green_confirm_elapsed += self.dt
      if self._green_confirm_elapsed >= E2E_GREEN_CONFIRM_S:
        self._green_confirmed = True
    elif standstill:
      self._green_confirm_elapsed = 0.0

    return bool(self._from_standstill and
                self._green_confirmed and
                not output_should_stop_mpc and
                not should_stop_e2e and
                output_a_target_e2e >= E2E_LAUNCH_BRAKE_YIELD)

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    launch_boost = self.chill_aggressive_launch.update(
      sm['carControl'].enabled, sm['selfdriveState'].experimentalMode, sm['selfdriveState'].personality,
      v_ego, v_cruise if v_cruise_initialized else 0., sm['carState'].brakePressed)
    accel_clip = [ACCEL_MIN, get_max_accel(v_ego, launch_boost)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    # Get new v_cruise and a_desired from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.a_desired = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.a_desired, v_cruise)

    if force_slow_decel:
      v_cruise = 0.0

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality,
                    experimental_mode=sm['selfdriveState'].experimentalMode,
                    cruise_max_accel=max(CRUISE_MAX_ACCEL, accel_clip[1]) if launch_boost else CRUISE_MAX_ACCEL)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                                                        action_t=action_t, vEgoStopping=self.CP.vEgoStopping)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    use_chill_launch = self._update_e2e_chill_launch(sm, v_ego, v_cruise, output_a_target_e2e,
                                                     output_should_stop_mpc, reset_state)
    if self.is_e2e(sm) and use_chill_launch:
      # Confirmed green after a stop: use chill/MPC accel up to set speed.
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc
    elif self.is_e2e(sm):
      output_a_target_e2e, output_should_stop_e2e = self._adjust_e2e_intersection_stop(
        sm, v_ego, output_a_target_e2e, output_should_stop_e2e)
      output_a_target = min(output_a_target_e2e, output_a_target_mpc)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc
      if output_a_target < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e
    else:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
