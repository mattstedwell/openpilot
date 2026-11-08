"""Chill/Aggressive longitudinal policies (speeds are SI except cruise output)."""
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


class ChillAggressiveLaunch:
  """Arm only at a stop, then taper back to stock by 10 m/s (~22 mph)."""
  def __init__(self):
    self.active = False

  def update(self, enabled, experimental_mode, personality, v_ego, v_cruise, brake_pressed=False):
    if (not enabled or not is_chill_aggressive(experimental_mode, personality) or brake_pressed or
        not math.isfinite(v_cruise) or v_cruise <= v_ego + .5 or v_ego >= 10.):
      self.active = False
    elif v_ego <= .5:
      self.active = True
    return self.active

  @staticmethod
  def accel_extra(v_ego):
    return .4 * (1. - min(max(v_ego, 0.), 10.) / 10.)
