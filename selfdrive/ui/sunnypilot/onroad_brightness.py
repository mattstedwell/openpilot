"""
Onroad brightness / Screen Off timer helpers.

These functions are intentionally free of UI toolkit imports so the touch-only
wake policy can be unit tested without starting the GUI.
"""
from enum import Enum, IntEnum


class OnroadBrightness(IntEnum):
  AUTO = 0
  AUTO_DARK = 1
  SCREEN_OFF = 2


class OnroadTimerStatus(Enum):
  NONE = 0
  PAUSE = 1
  RESUME = 2


ONROAD_BRIGHTNESS_TIMER_PAUSED = -1


def has_physical_touch(events) -> bool:
  return any(getattr(ev, "left_down", False) or getattr(ev, "left_pressed", False) or
             getattr(ev, "left_released", False) for ev in events)


def onroad_brightness_timer_expired(brightness: int, timer: int) -> bool:
  return brightness != OnroadBrightness.AUTO and timer == 0


def start_onroad_sleep_timer(brightness: int, timer_param: int, fps: int) -> int:
  if brightness == OnroadBrightness.AUTO:
    return 0
  if brightness == OnroadBrightness.AUTO_DARK:
    return 15 * fps
  return timer_param * fps


def tick_onroad_sleep_timer(timer: int) -> int:
  return timer - 1 if timer > 0 else timer


def apply_reset_onroad_sleep_timer(timer: int, timer_status: OnroadTimerStatus, brightness: int,
                                   timer_param: int, fps: int, touch: bool = False) -> int:
  """Update the onroad dimmer timer.

  NONE starts or restarts the inactivity delay (touch-wake call sites only).
  PAUSE freezes the timer so settings stay visible.
  RESUME unpauses when returning to the onroad view, but never wakes the
  display unless a physical touch is in the same frame (so ignition/start
  and inactivity-timeout returns stay dark).
  """
  if timer_status == OnroadTimerStatus.PAUSE and timer != ONROAD_BRIGHTNESS_TIMER_PAUSED:
    return ONROAD_BRIGHTNESS_TIMER_PAUSED

  if timer_status == OnroadTimerStatus.RESUME:
    if timer == ONROAD_BRIGHTNESS_TIMER_PAUSED:
      if touch:
        return start_onroad_sleep_timer(brightness, timer_param, fps)
      return 0
    return timer

  if (timer_param >= 0 and brightness != OnroadBrightness.AUTO and
      timer != ONROAD_BRIGHTNESS_TIMER_PAUSED):
    return start_onroad_sleep_timer(brightness, timer_param, fps)
  return timer


def apply_onroad_alert(timer: int, alert=None) -> int:
  # Visual events, including critical alerts, must never wake or extend the timer.
  del alert
  return tick_onroad_sleep_timer(timer)


def set_onroad_brightness(started: bool, awake: bool, brightness: int, timer: int,
                          cur_brightness: float) -> float:
  if not awake or not started:
    return cur_brightness

  if timer != 0:
    if brightness == OnroadBrightness.AUTO_DARK:
      return max(30.0, cur_brightness)
    return cur_brightness

  if brightness == OnroadBrightness.AUTO:
    return cur_brightness
  if brightness == OnroadBrightness.AUTO_DARK:
    return cur_brightness
  if brightness == OnroadBrightness.SCREEN_OFF:
    return 0.0

  # 3-22: 5% - 100%
  return float((brightness - 2) * 5)
