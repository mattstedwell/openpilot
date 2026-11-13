#!/usr/bin/env python3
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.accel_boost import AccelBoost
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.sunnypilot.selfdrive.car.always_chill import ALWAYS_CHILL, experimental_mode_active
from openpilot.sunnypilot.selfdrive.car.chill_aggressive import ChillAggressiveLaunch
from openpilot.sunnypilot.selfdrive.car import chill_relaxed

J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
J_CRUISE_BP = [0., 10.0, 25., 40.]
A_CRUISE_MIN = -0.5
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]


def get_cruise_accel(v_cruise, v_ego, a_cruise_prev, dt, cap=None):
  max_a = ACCEL_MAX if cap is None else min(float(cap), ACCEL_MAX)
  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, max_a)
  j_cruise = np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


def get_max_accel(v_ego, chill_aggressive_launch=False, chill_relaxed_mode=False):
  """Cruise accel ceiling for this tree.

  Official sunnypilot/dev stock is a flat ACCEL_MAX = 2.0 (plus the
  official get_cruise_accel jerk schedule), not the older speed-dependent
  [1.6, 1.2, 0.8, 0.6] curve. 'Standard stock' is that 2.0. Chill +
  Relaxed is 80% of it (1.6). Chill + Aggressive launch adds +0.4 at a
  stop, tapering out by 10 m/s, then clips at ACCEL_MAX — so on this
  tree the launch boost cannot exceed official 2.0 and is a no-op at
  the actuator. Report numbers against the C3X/old-test3 curve honestly.
  """
  stock = float(ACCEL_MAX)
  if chill_relaxed_mode:
    return stock * chill_relaxed.MAX_ACCEL_SCALE
  extra = ChillAggressiveLaunch.accel_extra(v_ego) if chill_aggressive_launch else 0.0
  return min(stock + extra, ACCEL_MAX)


def choose_acc_plan(a_mpc, source_mpc, stop_mpc, a_cruise, stop_cruise):
  """ACC / Chill longitudinal plan: min of lead-MPC and cruise only.

  Official sunnypilot/dev always also blends the model's
  desiredAcceleration (the e2e candidate), which is what stops for
  lights and signs and slows for curves. When ALWAYS_CHILL, that
  candidate is dropped — the same split as the pre-C4 official Chill
  path (MPC mode 'acc'). Model shouldStop is not used.
  """
  candidates = ((a_mpc, source_mpc, stop_mpc),
                (a_cruise, LongitudinalPlanSource.cruise, stop_cruise))
  a_target, source, _ = min(candidates, key=lambda c: c[0])
  return a_target, source, any(s for _, _, s in candidates)


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.accel_boost = AccelBoost()

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False
    self.chill_aggressive_launch = ChillAggressiveLaunch()

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    # Always-Chill: experimental_mode_active is False even if a stale
    # selfdriveState says otherwise, so these Chill-only policies apply.
    personality = sm['selfdriveState'].personality
    exp_mode = experimental_mode_active(getattr(getattr(sm['selfdriveState'], 'deprecated', None), 'experimentalMode', False))
    enabled = bool(sm['selfdriveState'].enabled)
    launch_boost = self.chill_aggressive_launch.update(
      enabled, exp_mode, personality,
      v_ego, v_cruise if v_cruise_initialized else 0., sm['carState'].brakePressed)
    chill_relaxed_mode = chill_relaxed.is_chill_relaxed(exp_mode, personality)
    cap = get_max_accel(v_ego, launch_boost, chill_relaxed_mode)

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, cap)
      self.a_cruise = self.output_a_target

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.output_a_target = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.output_a_target, v_cruise)

    self.mpc.set_weights(prev_accel_constraint, personality=personality,
                         experimental_mode=exp_mode)
    if self.CP.openpilotLongitudinalControl:
      self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    else:
      self.mpc.set_cur_state(v_ego, sm['carState'].aEgo)
    self.mpc.update(sm['radarState'], personality=personality, experimental_mode=exp_mode)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)

    self.a_cruise = get_cruise_accel(v_cruise, v_ego, self.a_cruise, self.dt, cap=cap)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    if ALWAYS_CHILL:
      # Permanent ACC / Chill: lead-MPC and cruise only. The model's
      # desiredAcceleration and shouldStop are not candidates, so lights,
      # signs, and model-planned stops do not slow or stop the car.
      output_a_target, self.mpc.source, self.output_should_stop = choose_acc_plan(
          output_a_target_mpc, self.mpc.source, output_should_stop_mpc,
          self.a_cruise, cruise_should_stop)
    else:
      output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
      output_should_stop_e2e = sm['modelV2'].action.shouldStop
      output_a_target_e2e = self.accel_boost.update(sm, output_a_target_e2e, output_a_target_mpc, self.a_cruise)
      candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                    (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop),
                    (output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e)]
      output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
      self.output_should_stop = any(s for _, _, s in candidates)

    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, cap)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.accelBoost = float(self.accel_boost.total_boost)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
