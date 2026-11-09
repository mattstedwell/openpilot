"""Toyota LTA button acts as SET: cruise target is the detected speed limit + 5 mph.

While active, the car's own target-speed suggestions (PCM SET_SPEED / UI_SET_SPEED, stalk
RES/+/-) are ignored. Cancel (or any PCM disengage / main off) turns the mode off and
clears the target; it stays off until the LTA button is pressed again.
Speeds in are SI (m/s); targets out are kph like v_cruise_kph.
"""
import math

from cereal import car
from openpilot.common.constants import CV

ButtonType = car.CarState.ButtonEvent.Type

LTA_SET_OFFSET_MS = 5 * CV.MPH_TO_MS  # 5 mph = 8.04672 km/h


def valid_speed_limit(speed_limit):
  return math.isfinite(speed_limit) and speed_limit > 0


class LtaSetTarget:
  def __init__(self):
    self.active = False
    self.target_kph = None
    self.cruise_enabled_prev = False

  def reset(self):
    self.active = False
    self.target_kph = None

  def update(self, button_events, cruise_available, cruise_enabled, speed_limit, minimum, maximum):
    set_pressed = any(b.type == ButtonType.setCruise and b.pressed for b in button_events)
    cancel_pressed = any(b.type == ButtonType.cancel for b in button_events)
    # Toyota TSS2 has no cancel-button signal on the bus; the PCM dropping cruise is the cancel.
    pcm_disengaged = self.cruise_enabled_prev and not cruise_enabled
    self.cruise_enabled_prev = cruise_enabled

    if cancel_pressed or pcm_disengaged or not cruise_available:
      self.reset()
      return None

    has_limit = valid_speed_limit(speed_limit)
    if set_pressed and has_limit:
      self.active = True
    # SET without a detected limit doesn't invent a speed; an active mode keeps its last target.

    if self.active and has_limit:
      self.target_kph = min(max((speed_limit + LTA_SET_OFFSET_MS) * CV.MS_TO_KPH, minimum), maximum)

    return self.target_kph if self.active else None
