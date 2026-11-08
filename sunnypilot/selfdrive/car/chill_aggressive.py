"""Chill/Aggressive cruise-target policy (speeds are SI except the cruise output)."""
import math

from cereal import log
from openpilot.common.constants import CV


def is_chill_aggressive(experimental_mode, personality):
  # Dynamic experimental control's temporary ACC state is NOT Chill mode.
  return not experimental_mode and personality == log.LongitudinalPersonality.aggressive


class SpeedLimitCruiseTarget:
  def __init__(self):
    self.enabled = False
    self.mode = False
    self.cancelled = False
    self.applied = False
    self.overridden = False

  def update_state(self, enabled, cancel, experimental_mode, personality):
    self.mode = is_chill_aggressive(experimental_mode, personality)
    if not enabled:
      self.cancelled = False
      self.applied = False
      self.overridden = False
    elif cancel:
      # Don't reapply while carControl is still catching up with the cancel.
      self.cancelled = True
    self.enabled = enabled and not self.cancelled

  def manual_speed_changed(self):
    if self.enabled and self.applied:
      self.overridden = True

  def target_kph(self, speed_limit, minimum, maximum):
    if not self.enabled or not self.mode or self.overridden or not math.isfinite(speed_limit) or speed_limit <= 0:
      return None
    # Use the resolver's raw limit, never its configurable/retained offset value.
    target = (speed_limit + 5 * CV.MPH_TO_MS) * CV.MS_TO_KPH
    self.applied = True
    return min(max(target, minimum), maximum)
