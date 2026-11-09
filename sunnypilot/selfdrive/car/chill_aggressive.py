"""Chill/Aggressive longitudinal policies (speeds are SI)."""
import math

from cereal import log


def is_chill_aggressive(experimental_mode, personality):
  # Dynamic experimental control's temporary ACC state is NOT Chill mode.
  return not experimental_mode and personality == log.LongitudinalPersonality.aggressive


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
