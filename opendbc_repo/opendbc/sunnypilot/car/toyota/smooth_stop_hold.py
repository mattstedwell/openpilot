"""Minimal last-mph / standstill-hold comfort for the Toyota PCM path.

longcontrol can already be smooth; what reaches the car is pcm_accel_cmd
after the Toyota PID. When the integrator is frozen at the pid→stopping
switch, a leftover I term is a step on ACCEL_CMD — the last-mph jerk
that a smooth longcontrol command does not by itself remove.

This fades that offset and rate-limits the sent command at 0.30 m/s³
versus the last sent value, only while stopping or in the last ~1 mph,
and only for non-urgent commands (above -1.5 m/s²). Urgent / emergency
keeps the official 4.0 m/s³ path. Approach braking (v ≳ 1 mph and not
in the stopping state) is not touched, so Relaxed approach and
slowdown stay as they are (t304u).

Not road-tested. panda and docs/SAFETY.md are untouched.
"""
from __future__ import annotations

import numpy as np

# Bae 2019 comfort-band floor; well under Müller 2013 ~1 m/s³ JND.
COMFORT_JERK = 0.30
URGENT_ACCEL = -1.5
# ~1.1 mph: last-mph window. Above this, and not in stopping, we
# pass the official PCM command through unchanged.
LAST_MPH_V = 0.5


def _finite(*xs) -> bool:
  return all(x is not None and np.isfinite(x) for x in xs)


def smooth_pcm_accel(pcm_after_pid, actuators_accel, last_sent, dt_tick,
                     stopping, v_ego, long_active):
  """ACCEL_CMD after the PID, for one 3-frame Toyota long tick.

  Non-urgent stopping / last-mph: drop the frozen-I offset (follow
  the chauffeur actuators.accel) and rate-limit at COMFORT_JERK vs
  the last sent value. Everything else is returned unchanged.
  """
  if not long_active:
    return pcm_after_pid
  in_hold = bool(stopping)
  if (not in_hold) and _finite(v_ego) and float(v_ego) < LAST_MPH_V:
    in_hold = True
  if not in_hold:
    return pcm_after_pid

  # Urgent / emergency: official path (already 4.0-limited before PID).
  if _finite(actuators_accel) and float(actuators_accel) <= URGENT_ACCEL:
    return pcm_after_pid
  if _finite(pcm_after_pid) and float(pcm_after_pid) <= URGENT_ACCEL:
    return pcm_after_pid

  if _finite(actuators_accel):
    target = float(actuators_accel)
  elif _finite(pcm_after_pid):
    target = float(pcm_after_pid)
  else:
    return pcm_after_pid

  # Launch / positive creep: never clamp a go.
  if target > 0.0:
    return pcm_after_pid

  if not _finite(dt_tick) or dt_tick <= 0.0:
    return target
  if not _finite(last_sent):
    return target
  step = COMFORT_JERK * float(dt_tick)
  return float(np.clip(target, float(last_sent) - step, float(last_sent) + step))


def fade_stopping_integral(pid, dt_tick, rate=COMFORT_JERK):
  """Unwind a frozen I term at the comfort jerk, only while stopping.

  The sent command already ignores I (smooth_pcm_accel follows
  actuators.accel). This just clears the leftover so a later pid
  resume is not offset. Official ACCEL_PID_UNWIND is 0.03 m/s²/s;
  this is faster, but only in the stopping state.
  """
  if pid is None:
    return
  i = getattr(pid, "i", 0.0)
  if not _finite(i) or i == 0.0:
    return
  if not _finite(dt_tick) or dt_tick <= 0.0:
    return
  step = float(rate) * float(dt_tick)
  if abs(float(i)) <= step:
    pid.i = 0.0
  else:
    pid.i = float(i) - step * (1.0 if i > 0.0 else -1.0)
