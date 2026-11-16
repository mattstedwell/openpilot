#!/usr/bin/env python3
import math
import numpy as np
from types import SimpleNamespace

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
from openpilot.sunnypilot.selfdrive.controls.lib.lite_launch import PRE_HOLD, LiteLaunch, stop_target as lite_stop_target
from openpilot.sunnypilot.selfdrive.controls.lib.lite_e2e_stop import E2eStopState, planned_stop_x, shift_e2e_stop
from openpilot.sunnypilot.selfdrive.controls.lib.lite_lead_holdover import LeadHoldover
from openpilot.sunnypilot.selfdrive.controls.lib.lite_launch_boost import LiteLaunchBoost
from openpilot.sunnypilot.selfdrive.controls.lib.lite_lead_lock import LeadLock
from openpilot.sunnypilot.selfdrive.controls.lib.lite_gap_open import LiteGapOpen
from openpilot.sunnypilot.selfdrive.controls.lib.lite_turn_cap import cap_accel as turn_cap_accel, curvature_from_angle

J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
J_CRUISE_BP = [0., 10.0, 25., 40.]
A_CRUISE_MIN = -0.5
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]


def get_cruise_accel(v_cruise, v_ego, a_cruise_prev, dt):
  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, ACCEL_MAX)
  j_cruise = np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.accel_boost = AccelBoost()
    self.lite_launch = LiteLaunch()
    self.launch_boost = LiteLaunchBoost()
    self.e2e_stop_state = E2eStopState()
    self.lead_lock = LeadLock()
    self.gap_open = LiteGapOpen()
    self.vm = None  # VehicleModel for the turn felt-g cap (lazy)
    self.lead_holdover = LeadHoldover()

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False

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

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.output_a_target = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.output_a_target, v_cruise)

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    if self.CP.openpilotLongitudinalControl:
      self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    else:
      self.mpc.set_cur_state(v_ego, sm['carState'].aEgo)
    # dev-c4-sunnypilot-lite: keep a decelerating, never-accelerating phantom
    # lead for <= 1 s if radar drops leadOne (no FCW from it). MPC only.
    radar_state = sm['radarState']
    if reset_state:
      self.lead_holdover.reset()
    lead_one = self.lead_holdover.update(radar_state.leadOne, v_ego, self.dt)
    if lead_one is not radar_state.leadOne:
      radar_state = SimpleNamespace(leadOne=lead_one, leadTwo=radar_state.leadTwo)
    self.mpc.update(radar_state, personality=sm['selfdriveState'].personality)

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
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop
    model_stop_x = planned_stop_x(sm['modelV2'])  # the model's own planned stop (before the lite 10 ft shift)
    # dev-c4-sunnypilot-lite: stop ~10 ft before the model's planned stop at
    # lights / stop signs (skipped when a lead explains the stop; the lead
    # MPC already adds 10 ft there). Can only lower accel or add shouldStop.
    output_a_target_e2e, output_should_stop_e2e = shift_e2e_stop(sm['modelV2'], sm['radarState'].leadOne, v_ego,
                                                                 output_a_target_e2e, output_should_stop_e2e,
                                                                 state=self.e2e_stop_state, dt=self.dt)

    self.a_cruise = get_cruise_accel(v_cruise, v_ego, self.a_cruise, self.dt)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    output_a_target_e2e = self.accel_boost.update(sm, output_a_target_e2e, output_a_target_mpc, self.a_cruise)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop),
                  (output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e)]

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)

    # dev-c4-sunnypilot-lite: quicker start once the lead clearly pulls away.
    # Gates re-checked every cycle, nothing latched; see lite_launch.py.
    # Vision + radar: the model's own lead is only used to AGREE before going
    # sooner / harder (lite_fusion); stops and holds use leadOne as before.
    model_leads = sm['modelV2'].leadsV3
    model_lead = model_leads[0] if len(model_leads) > 0 else None
    if reset_state:
      self.lite_launch.reset()
      self.launch_boost.reset()
      self.e2e_stop_state.reset()
      self.lead_lock.reset()
      self.gap_open.reset()
    # sunnypilot-lite-test-fix: radar-only launch. The model's e2e stop no longer
    # holds the launch unless it is a stop the lead does not explain (line /
    # sign / light, lite_fusion.model_stop_blocks); the "plan not braking" gate
    # uses the stock plan without the model candidate.
    a_plan_radar = min(output_a_target_mpc, self.a_cruise)
    quick_start, output_a_target = self.lite_launch.update(sm['radarState'].leadOne, v_ego, sm['carState'].standstill,
                                                           self.fcw, output_a_target, output_a_target_e2e,
                                                           output_should_stop_e2e, model_lead=model_lead,
                                                           model_stop_x=model_stop_x, a_plan_radar=a_plan_radar, dt=self.dt)
    if quick_start:
      self.output_should_stop = False
    elif self.lite_launch.pending:
      # pre-release: light brake hold (still braking, no motion) while the radar lead starts to roll
      self.output_should_stop = False
      output_a_target = PRE_HOLD if output_a_target <= 0.0 else output_a_target
    # sunnypilot-lite launchboost: stronger take-up behind a lead pulling away
    # from a stop (floor only; gates and fades in lite_launch_boost.py).
    boost_on, output_a_target = self.launch_boost.update(sm['radarState'].leadOne, v_ego, sm['carState'].standstill, self.fcw,
                                                         sm['selfdriveState'].personality, output_a_target, a_prev, self.dt,
                                                         output_a_target_e2e, output_should_stop_e2e, model_lead=model_lead,
                                                         model_stop_x=model_stop_x, a_plan_radar=a_plan_radar)
    if boost_on:
      self.output_should_stop = False
    # stop-and-go lead lock: right after both cars stop, up to 5 mph, follow the radar lead's accel
    # (feed-forward) and hold the stopped gap; hard safety overrides in lite_lead_lock.py.
    lock_on, output_a_target = self.lead_lock.update(sm['radarState'].leadOne, v_ego, sm['carState'].standstill, self.fcw,
                                                     output_a_target, a_plan_radar,
                                                     launch_active=bool(quick_start or boost_on),
                                                     dt=self.dt, stop_target=lite_stop_target())
    if lock_on and output_a_target > 0.0:
      self.output_should_stop = False
    # after a standstill launch, softly ease off under the lead if the gap is still
    # below the follow distance. The under-lead offset ramps in from 2.5 mph (full by
    # 6.5), and shrinks smoothly to 0 as the gap reaches 100% of target. Applied as
    # min() on top of lock/boost (never raises accel, never weakens braking, drops if
    # the lead slows). Turn cap stays last.
    _, output_a_target = self.gap_open.update(sm['radarState'].leadOne, v_ego, sm['carState'].standstill, self.fcw,
                                              sm['selfdriveState'].personality, output_a_target, a_prev, self.dt,
                                              lock_active=bool(lock_on), boost_active=bool(boost_on))
    # no hard accel into low-speed intersection turns (steering angle + lat accel), after every launch path
    output_a_target = turn_cap_accel(output_a_target, v_ego, self.turn_curvature(sm['carState']),
                                     sm['carState'].steeringAngleDeg, sm['carState'].steeringRateDeg)
    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def turn_curvature(self, CS) -> float:
    """Path curvature (1/m) from the steering wheel angle (VehicleModel when available)."""
    try:
      if self.vm is None:
        from opendbc.car.vehicle_model import VehicleModel
        self.vm = VehicleModel(self.CP)
      return float(self.vm.calc_curvature(math.radians(CS.steeringAngleDeg), max(CS.vEgo, 0.1), 0.0))
    except Exception:
      return curvature_from_angle(CS.steeringAngleDeg)

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
