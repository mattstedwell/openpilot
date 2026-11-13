"""Permanent for this build: longitudinal always runs in Chill (ACC) mode.

The model's e2e desiredAcceleration and shouldStop are never candidates
in the planner. Official Experimental mode (the three-way min of
lead-MPC, cruise, and e2e) is restored by setting ALWAYS_CHILL False
and building again. This tree has no ExperimentalMode param or Settings
toggle; the cereal field is deprecated and is forced off.
"""
ALWAYS_CHILL = True


def experimental_mode_active(requested: bool = False) -> bool:
  """What selfdriveState.deprecated.experimentalMode may report.

  On this build the requested value is ignored; the result is always False.
  """
  if ALWAYS_CHILL:
    return False
  return bool(requested)
