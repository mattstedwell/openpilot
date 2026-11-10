from types import SimpleNamespace

import pytest

from openpilot.selfdrive.ui.sunnypilot.onroad_brightness import (
  ONROAD_BRIGHTNESS_TIMER_PAUSED, OnroadBrightness, OnroadTimerStatus,
  apply_onroad_alert, apply_reset_onroad_sleep_timer, has_physical_touch,
  onroad_brightness_timer_expired, set_onroad_brightness, start_onroad_sleep_timer,
  tick_onroad_sleep_timer,
)

FPS = 20
DELAY_S = 3
DELAY_FRAMES = DELAY_S * FPS


def touch_event(*, down=False, pressed=False, released=False):
  return SimpleNamespace(left_down=down, left_pressed=pressed, left_released=released)


def reset(timer, status=OnroadTimerStatus.NONE, brightness=OnroadBrightness.SCREEN_OFF,
          timer_param=DELAY_S, fps=FPS, touch=False):
  return apply_reset_onroad_sleep_timer(timer, status, brightness, timer_param, fps, touch=touch)


def brightness(timer, started=True, awake=True, mode=OnroadBrightness.SCREEN_OFF, cur=80.0):
  return set_onroad_brightness(started, awake, mode, timer, cur)


# Events that previously woke the screen and must now leave it dark.
ALERT_CASES = [
  SimpleNamespace(alert_type="fcw/warning", text1="BRAKE"),
  SimpleNamespace(alert_type="hardBraking/critical", text1="BRAKE!"),
  SimpleNamespace(alert_type="stockFcw/warning", text1="BRAKE"),
  SimpleNamespace(alert_type="ldw/warning", text1="KEEP HANDS ON WHEEL"),
  SimpleNamespace(alert_type="preLaneChangeLeft/warning", text1="Lane Change"),
  SimpleNamespace(alert_type="preLaneChangeRight/warning", text1="Lane Change"),
  SimpleNamespace(alert_type="laneChange/warning", text1="Lane Change"),
  SimpleNamespace(alert_type="laneChangeBlocked/warning", text1="Blocked"),
  SimpleNamespace(alert_type="speedLimitPreActive/warning", text1="Speed Limit"),
  SimpleNamespace(alert_type="steerRequired/critical", text1="TAKE CONTROL"),
  "greenLightAlert",
  "leadDepartAlert",
  "leftBlinker",
  "rightBlinker",
]


def test_defaults_start_dark_after_init():
  # Timer starts at 0; going onroad after init must stay dark without a touch.
  timer = 0
  timer = reset(timer, OnroadTimerStatus.RESUME, touch=False)
  assert timer == 0
  assert onroad_brightness_timer_expired(OnroadBrightness.SCREEN_OFF, timer)
  assert brightness(timer) == 0.0


def test_ignition_start_transition_does_not_wake():
  # started/ignition becoming true triggers onroad show_event (RESUME) with no touch.
  timer = 0
  assert brightness(timer, started=False) == 80.0  # offroad/boot UI stays lit
  timer = reset(timer, OnroadTimerStatus.RESUME, touch=False)
  assert timer == 0
  assert brightness(timer, started=True) == 0.0


@pytest.mark.parametrize("alert", ALERT_CASES)
def test_visual_events_do_not_wake_dark_screen(alert):
  timer = apply_onroad_alert(0, alert)
  assert timer == 0
  assert brightness(timer) == 0.0
  # NONE reset is the old event-wake path; call sites must not use it for alerts,
  # but a stray resume without touch also must not light the display.
  assert reset(0, OnroadTimerStatus.RESUME, touch=False) == 0


@pytest.mark.parametrize("alert", ALERT_CASES)
def test_visual_events_do_not_extend_lit_timer(alert):
  timer = DELAY_FRAMES
  after_alert = apply_onroad_alert(timer, alert)
  after_tick = tick_onroad_sleep_timer(timer)
  assert after_alert == after_tick == DELAY_FRAMES - 1
  assert after_alert != DELAY_FRAMES


def test_touch_wakes_and_existing_inactivity_returns_to_dark():
  timer = 0
  assert brightness(timer) == 0.0
  assert has_physical_touch([touch_event(down=True)])

  # A physical touch starts the existing Onroad Brightness Delay (default 3s).
  timer = reset(timer, OnroadTimerStatus.NONE, touch=True)
  assert timer == DELAY_FRAMES
  assert brightness(timer) == 80.0
  assert not onroad_brightness_timer_expired(OnroadBrightness.SCREEN_OFF, timer)

  for _ in range(DELAY_FRAMES):
    timer = tick_onroad_sleep_timer(timer)
  assert timer == 0
  assert brightness(timer) == 0.0

  # A later touch wakes the display again.
  timer = reset(timer, OnroadTimerStatus.NONE, touch=True)
  assert timer == DELAY_FRAMES
  assert brightness(timer) == 80.0


def test_inactivity_delay_uses_setting():
  assert start_onroad_sleep_timer(OnroadBrightness.SCREEN_OFF, 3, FPS) == 60
  assert start_onroad_sleep_timer(OnroadBrightness.SCREEN_OFF, 0, FPS) == 0
  # A 0s delay expires immediately, so the display stays dark after the touch frame.
  timer = reset(0, OnroadTimerStatus.NONE, timer_param=0)
  assert timer == 0
  assert brightness(timer) == 0.0


def test_settings_pause_stays_visible_timeout_return_stays_dark():
  timer = reset(0, OnroadTimerStatus.NONE)
  assert brightness(timer) == 80.0

  paused = reset(timer, OnroadTimerStatus.PAUSE)
  assert paused == ONROAD_BRIGHTNESS_TIMER_PAUSED
  assert brightness(paused) == 80.0

  # Inactivity timeout closes settings with no touch: must go dark immediately.
  timed_out = reset(paused, OnroadTimerStatus.RESUME, touch=False)
  assert timed_out == 0
  assert brightness(timed_out) == 0.0


def test_settings_close_with_touch_restarts_inactivity_delay():
  paused = reset(DELAY_FRAMES, OnroadTimerStatus.PAUSE)
  assert has_physical_touch([touch_event(released=True)])
  timer = reset(paused, OnroadTimerStatus.RESUME, touch=True)
  assert timer == DELAY_FRAMES
  assert brightness(timer) == 80.0


def test_physical_touch_detection():
  assert not has_physical_touch([])
  assert not has_physical_touch([touch_event()])
  assert has_physical_touch([touch_event(down=True)])
  assert has_physical_touch([touch_event(pressed=True)])
  assert has_physical_touch([touch_event(released=True)])


def test_auto_mode_unaffected():
  timer = reset(0, OnroadTimerStatus.NONE, brightness=OnroadBrightness.AUTO)
  assert timer == 0
  assert brightness(timer, mode=OnroadBrightness.AUTO, cur=55.0) == 55.0
  assert apply_onroad_alert(0, ALERT_CASES[0]) == 0
