"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from cereal import messaging, log, car, custom
from openpilot.common.params import Params
from openpilot.selfdrive.ui.sunnypilot.onroad_brightness import (
  OnroadBrightness, OnroadTimerStatus,
  apply_onroad_alert, apply_reset_onroad_sleep_timer, has_physical_touch,
  onroad_brightness_timer_expired, set_onroad_brightness as compute_onroad_brightness,
)
from openpilot.sunnypilot.sunnylink.sunnylink_state import SunnylinkState
from openpilot.system.ui.lib.application import gui_app

OpenpilotState = log.SelfdriveState.OpenpilotState
MADSState = custom.ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState


class UIStateSP:
  def __init__(self):
    self.params = Params()
    self.CP_SP: custom.CarParamsSP | None = None
    self.has_icbm: bool = False
    self.is_sp_release: bool = self.params.get_bool("IsReleaseSpBranch")
    self.sm_services_ext = [
      "modelManagerSP", "selfdriveStateSP", "longitudinalPlanSP", "backupManagerSP",
      "gpsLocation", "liveTorqueParameters", "carStateSP", "liveMapDataSP", "carParamsSP", "liveDelay"
    ]

    self.sunnylink_state = SunnylinkState()

    self.onroad_brightness_timer: int = 0
    self.custom_interactive_timeout: int = 0

  def update(self) -> None:
    if self.sunnylink_enabled:
      self.sunnylink_state.start()
    else:
      self.sunnylink_state.stop()

  def onroad_brightness_handle_alerts(self, _ui_state, alert):
    # Visual alerts must never wake the display. Only tick the inactivity timer.
    if _ui_state.sm.recv_frame["carState"] < _ui_state.started_frame:
      return
    self.onroad_brightness_timer = apply_onroad_alert(self.onroad_brightness_timer, alert)

  def update_onroad_brightness(self, has_alert: bool) -> None:
    # Kept for compatibility; alerts no longer pause or restart the countdown.
    del has_alert
    self.onroad_brightness_timer = apply_onroad_alert(self.onroad_brightness_timer)

  def reset_onroad_sleep_timer(self, timer_status: OnroadTimerStatus = OnroadTimerStatus.NONE) -> None:
    touch = has_physical_touch(gui_app.mouse_events) if timer_status == OnroadTimerStatus.RESUME else False
    self.onroad_brightness_timer = apply_reset_onroad_sleep_timer(
      self.onroad_brightness_timer, timer_status, self.onroad_brightness,
      self.onroad_brightness_timer_param, gui_app.target_fps, touch=touch,
    )

  @property
  def onroad_brightness_timer_expired(self) -> bool:
    return onroad_brightness_timer_expired(self.onroad_brightness, self.onroad_brightness_timer)

  @property
  def auto_onroad_brightness(self) -> bool:
    return self.onroad_brightness in (OnroadBrightness.AUTO, OnroadBrightness.AUTO_DARK)

  @staticmethod
  def update_status(ss, ss_sp, onroad_evt) -> str:
    state = ss.state
    mads = ss_sp.mads
    mads_state = mads.state

    if state == OpenpilotState.preEnabled:
      return "override"

    if state == OpenpilotState.overriding:
      if not mads.available:
        return "override"

      if any(e.overrideLongitudinal for e in onroad_evt):
        return "override"

    if mads_state in (MADSState.paused, MADSState.overriding):
      return "override"

    # MADS specific statuses
    if not mads.available:
      return "engaged" if ss.enabled else "disengaged"

    if not mads.enabled and not ss.enabled:
      return "disengaged"

    if mads.enabled and ss.enabled:
      return "engaged"

    if mads.enabled:
      return "lat_only"

    if ss.enabled:
      return "long_only"

    return "disengaged"

  def update_params(self) -> None:
    CP_SP_bytes = self.params.get("CarParamsSPPersistent")
    if CP_SP_bytes is not None:
      self.CP_SP = messaging.log_from_bytes(CP_SP_bytes, custom.CarParamsSP)
      self.has_icbm = self.CP_SP.intelligentCruiseButtonManagementAvailable and self.params.get_bool("IntelligentCruiseButtonManagement")

    self._enforce_constraints()
    self.active_bundle = self.params.get("ModelManager_ActiveBundle")
    self.blindspot = self.params.get_bool("BlindSpot")
    self.chevron_metrics = self.params.get("ChevronInfo")
    self.custom_interactive_timeout = self.params.get("InteractivityTimeout", return_default=True)
    self.developer_ui = self.params.get("DevUIInfo")
    self.hide_v_ego_ui = self.params.get_bool("HideVEgoUI")
    self.onroad_brightness = int(float(self.params.get("OnroadScreenOffBrightness", return_default=True)))
    self.onroad_brightness_timer_param = self.params.get("OnroadScreenOffTimer", return_default=True)
    self.rainbow_path = self.params.get_bool("RainbowMode")
    self.road_name_toggle = self.params.get_bool("RoadNameToggle")
    self.rocket_fuel = self.params.get_bool("RocketFuel")
    self.speed_limit_mode = self.params.get("SpeedLimitMode", return_default=True)
    self.standstill_timer = self.params.get_bool("StandstillTimer")
    self.sunnylink_enabled = self.params.get_bool("SunnylinkEnabled")
    self.torque_bar = self.params.get_bool("TorqueBar")
    self.enforce_torque_control = self.params.get_bool("EnforceTorqueControl")
    self.custom_torque_params = self.params.get_bool("CustomTorqueParams")
    self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")
    self.torque_override_lat_accel_factor = float(self.params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
    self.torque_override_friction = float(self.params.get("TorqueParamsOverrideFriction", return_default=True))
    self.true_v_ego_ui = self.params.get_bool("TrueVEgoUI")
    self.turn_signals = self.params.get_bool("ShowTurnSignals")
    self.boot_offroad_mode = self.params.get("DeviceBootMode", return_default=True)
    self.always_offroad = self.params.get_bool("OffroadMode")

  def _enforce_constraints(self) -> None:
    has_long = self.has_longitudinal_control
    CP = self.CP

    if CP is not None:
      # Angle steering: no torque-based lateral controls
      if CP.steerControlType == car.CarParams.SteerControlType.angle:
        self.params.remove("EnforceTorqueControl")
        self.params.remove("NeuralNetworkLateralControl")

      # Alpha longitudinal: clear if not available
      if not CP.alphaLongitudinalAvailable:
        self.params.remove("AlphaLongitudinalEnabled")

      # BSM not available: clear BSM-dependent settings
      if not CP.enableBsm:
        self.params.remove("AutoLaneChangeBsmDelay")
    else:
      # No CarParams: clear all car-dependent params as safety default
      self.params.remove("EnforceTorqueControl")
      self.params.remove("NeuralNetworkLateralControl")
      self.params.remove("AlphaLongitudinalEnabled")

    # No longitudinal control: no experimental mode or DEC
    if not has_long:
      self.params.remove("ExperimentalMode")
      self.params.remove("DynamicExperimentalControl")

    # ICBM: clear if not available or if full longitudinal control is active
    if self.CP_SP is not None:
      if not self.CP_SP.intelligentCruiseButtonManagementAvailable or has_long:
        self.params.remove("IntelligentCruiseButtonManagement")
        self.has_icbm = False
    else:
      self.params.remove("IntelligentCruiseButtonManagement")
      self.has_icbm = False

    # Cruise features requiring longitudinal or ICBM
    if not (has_long or self.has_icbm):
      self.params.remove("CustomAccIncrementsEnabled")
      self.params.remove("SmartCruiseControlVision")
      self.params.remove("SmartCruiseControlMap")


class DeviceSP:
  @staticmethod
  def _set_awake(on: bool, _ui_state):
    if _ui_state.boot_offroad_mode == 1 and not on:
      _ui_state.params.put_bool("OffroadMode", True)

  @staticmethod
  def set_onroad_brightness(_ui_state, awake: bool, cur_brightness: float) -> float:
    return compute_onroad_brightness(
      _ui_state.started, awake, _ui_state.onroad_brightness,
      _ui_state.onroad_brightness_timer, cur_brightness,
    )

  @staticmethod
  def set_min_onroad_brightness(_ui_state, min_brightness: int) -> int:
    if _ui_state.onroad_brightness == OnroadBrightness.AUTO_DARK:
      min_brightness = 10

    return min_brightness

  @staticmethod
  def wake_from_dimmed_onroad_brightness(_ui_state, evs) -> None:
    if _ui_state.started and (_ui_state.onroad_brightness_timer_expired or _ui_state.onroad_brightness == OnroadBrightness.AUTO_DARK):
      if any(ev.left_down for ev in evs):
        if _ui_state.onroad_brightness_timer_expired:
          gui_app.mouse_events.clear()
        _ui_state.reset_onroad_sleep_timer()
