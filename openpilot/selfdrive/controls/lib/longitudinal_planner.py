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
from openpilot.sunnypilot.selfdrive.car.chill_aggressive import is_chill_aggressive
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_accel_curve import limit_accel_rise, smooth_accel_cap
from openpilot.sunnypilot.selfdrive.car import chill_relaxed
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_curve_comfort import CurveComfort
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_safe_launch import (
  SafeLaunch, desired_follow_gap, launch_gate_ok)
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_sng import StopAndGo
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_jerk_factor, get_T_FOLLOW, LEAD_DANGER_FACTOR

# dev-c4-sng-test: horizon (first ~2 s) used for the urgency check
SNG_PLAN_N = int(np.searchsorted(T_IDXS_MPC, 2.0)) + 1

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


# Previous C3X / old test3 piecewise-linear cruise accel curve, kept for
# reference and before/after reports only (t314u replaced it with
# devc4_test3_accel_curve.smooth_accel_cap).
A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
HWY_EQUAL_V_START = 35. * CV.MPH_TO_MS  # m/s; Agg/Std start blending to Relaxed's cap (t316u)
HWY_EQUAL_V_FULL = 50. * CV.MPH_TO_MS   # m/s; Agg/Std cap == Relaxed's at and above (t316u)


def get_max_accel(v_ego, aggressive=False, chill_relaxed_mode=False):
  """Positive cruise accel ceiling (m/s²), devc4-test3 t314u.

  One smooth, monotone (PCHIP, C1) curve per personality, a bit gentler
  than the old C3X numbers and falling with speed like a fixed fraction
  of power. Aggressive / Standard blend onto Relaxed's curve over
  35-50 mph (smootherstep, C2; t316u) and equal it at >= 50 mph (t312u). Aggressive's
  low-speed extra is built into its own curve. Applied as a cap on
  get_cruise_accel and on the chosen plan; braking is not capped here.
  """
  if chill_relaxed_mode:
    name = "relaxed"
  elif aggressive:
    name = "aggressive"
  else:
    name = "standard"
  return min(smooth_accel_cap(v_ego, name), ACCEL_MAX)


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
    self.curve_comfort = CurveComfort()
    self.safe_launch = SafeLaunch()
    self.sng = StopAndGo()

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
    chill_relaxed_mode = chill_relaxed.is_chill_relaxed(exp_mode, personality)
    cap = get_max_accel(v_ego, is_chill_aggressive(exp_mode, personality), chill_relaxed_mode)
    # devc4-test3 curve comfort (t311u): lower only the positive accel
    # ceiling in curves (current + model-planned lateral accel). Braking
    # is never limited: cap is only the upper clip and stays >= 0.
    if reset_state:
      self.curve_comfort.reset()
    md = sm['modelV2']
    cap = self.curve_comfort.update(
      cap, v_ego, sm['controlsState'].curvature,
      md.orientationRate.z, md.velocity.x, ModelConstants.T_IDXS, self.dt)

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

    # devc4-test3 safe launch (t309u): keep official standstill MPC weighting
    # for the first ~5 mph behind a lead that is clearly pulling away (all
    # gates), and for the rest of that launch while the lead slows / we
    # close under ~10 mph so braking builds sooner. Accel is still the MPC
    # plan; nothing is added to it. self.fcw is the previous frame's.
    # dev-c4-sng-test: off by default (SngConfig.safe_launch); the quicker
    # start comes from the Toyota brake-hold release in opendbc instead.
    lead_one = sm['radarState'].leadOne
    desired_gap = desired_follow_gap(v_ego, getattr(lead_one, 'vLead', float('nan')), personality, exp_mode)
    if reset_state:
      self.safe_launch.reset()
      self.sng.reset()
    if self.sng.config.safe_launch:
      if self.sng.config.safe_launch_brake_free:
        prev_accel_constraint = self.safe_launch.prev_accel_constraint(
          prev_accel_constraint, v_ego, lead_one, desired_gap, self.fcw)
      else:
        # launch gates only (no change-cost drop for the rest of the launch)
        prev_accel_constraint = prev_accel_constraint and not launch_gate_ok(v_ego, lead_one, desired_gap, self.fcw)

    # dev-c4-sng-test: automatic stop-and-go / city smoothing (Chill only).
    # MPC inputs only: Relaxed-style jerk factor (faded out when braking is
    # urgent), stop gap planned from the lead's predicted stop, higher lead
    # danger factor / danger-zone cost (keep the set gap), and a lower
    # positive accel ceiling in stop-and-go. The plan is never raised and
    # braking is never limited; see devc4_sng.py.
    jerk_factor, t_follow, danger_cost = None, None, None
    self.mpc.stop_extra_predicted = 0.0
    self.mpc.lead_danger_factor = LEAD_DANGER_FACTOR
    if not exp_mode:
      sng = self.sng.update(v_ego, lead_one, get_jerk_factor(personality, exp_mode), cap,
                            smooth_accel_cap(v_ego, "relaxed"), desired_gap, self.output_a_target, self.fcw, self.dt,
                            base_t_follow=get_T_FOLLOW(personality, exp_mode, self.v_desired_filter.x),
                            plan_a_min=float(np.min(self.mpc.a_solution[:SNG_PLAN_N])))
      cap = min(cap, sng.cap)
      jerk_factor, t_follow, danger_cost = sng.jerk_factor, sng.t_follow, sng.danger_cost
      self.mpc.stop_extra_predicted = sng.stop_extra_predicted
      if sng.danger_factor is not None:
        self.mpc.lead_danger_factor = sng.danger_factor
      if sng.brake_free:
        prev_accel_constraint = False

    self.mpc.set_weights(prev_accel_constraint, personality=personality,
                         experimental_mode=exp_mode, jerk_factor=jerk_factor, danger_cost=danger_cost)
    if self.CP.openpilotLongitudinalControl:
      self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    else:
      self.mpc.set_cur_state(v_ego, sm['carState'].aEgo)
    self.mpc.update(sm['radarState'], personality=personality, experimental_mode=exp_mode, t_follow=t_follow)

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
    # t314u: smooth transitions (cruise resume, lead leaving, speed-up):
    # limit only increases of positive accel (0.6 -> 0.45 m/s³ by speed).
    # Less accel / more braking is never delayed.
    if not reset_state:
      self.output_a_target = limit_accel_rise(self.output_a_target, a_prev, v_ego, self.dt)

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
