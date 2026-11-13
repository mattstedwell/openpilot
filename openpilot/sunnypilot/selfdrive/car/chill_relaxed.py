"""Chill/Relaxed longitudinal comfort overlay (speeds are SI).

Relaxed is the highway profile. In Chill mode (Experimental Mode off) it
follows farther back at speed, accelerates more gently, and smooths its
accel changes. Standard, Aggressive, and Experimental Mode are unchanged
relative to this tree's official stock.
"""
import numpy as np

from openpilot.cereal import log

# Follow time: stock Relaxed 1.75 s until ~30 mph, ramping to 2.15 s by ~50 mph,
# so stop-and-go stays as before and the highway gap grows by ~0.4 s.
T_FOLLOW_BP = [13.4, 22.4]  # m/s (~30 mph, ~50 mph)
T_FOLLOW_V = [1.75, 2.15]   # s
# Fraction of this tree's official cruise accel ceiling (ACCEL_MAX = 2.0).
MAX_ACCEL_SCALE = 0.8
# Multiplies the jerk/accel-change costs (stock Relaxed is 1.0).
JERK_FACTOR = 1.3


def is_chill_relaxed(experimental_mode, personality):
  # Dynamic experimental control's temporary ACC state is NOT Chill mode.
  return not experimental_mode and personality == log.LongitudinalPersonality.relaxed


def t_follow(v_ego):
  return float(np.interp(v_ego, T_FOLLOW_BP, T_FOLLOW_V))
