"""Unit tests for the t309u safe launch gates (no solver needed)."""
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_safe_launch import (
  SafeLaunch, brake_response_free, launch_gate_ok,
)
from opendbc.sunnypilot.car.toyota.launch_release import LAUNCH_RELEASE_HOLD, release_prev_accel

DESIRED = 7.524


def lead(d=7.7, v=0.5, a=0.8, present=True):
  return SimpleNamespace(status=present, present=present, dRel=d, vLead=v, aLeadK=a)


def test_gate_passes_for_clean_departure():
  assert launch_gate_ok(0.0, lead(), DESIRED)


@pytest.mark.parametrize("kw,why", [
  (dict(a=-0.2), "lead slowing"),
  (dict(v=0.05, a=0.0), "lead ~stopped"),
  (dict(v=0.0, a=0.0), "lead stopped"),
  (dict(d=DESIRED), "gap at desired"),
  (dict(d=DESIRED - 0.5), "gap below desired"),
  (dict(present=False), "no lead"),
  (dict(d=float("nan")), "bad radar"),
])
def test_gate_blocks(kw, why):
  assert not launch_gate_ok(0.0, lead(**kw), DESIRED), why


def test_gate_blocks_closing_fcw_and_above_5mph():
  assert not launch_gate_ok(0.6, lead(v=0.5), DESIRED)          # closing (v_ego >= v_lead)
  assert not launch_gate_ok(0.0, lead(), DESIRED, fcw=True)
  assert not launch_gate_ok(2.5, lead(v=4.0, d=20.0), DESIRED)  # > ~5 mph


def test_brake_free_only_when_slowing_closing_or_close():
  assert brake_response_free(1.5, lead(v=1.4, a=0.3), DESIRED)          # closing
  assert brake_response_free(1.5, lead(v=2.0, a=-0.5, d=9.0), DESIRED)  # lead slowing
  assert brake_response_free(1.5, lead(v=2.0, a=0.3, d=7.0), 9.0)       # under desired
  assert not brake_response_free(1.5, lead(v=2.0, a=0.3, d=12.0), 9.0)  # clean follow
  assert not brake_response_free(0.0, lead(v=1.4, a=-1.0), DESIRED)     # standstill: official
  assert not brake_response_free(6.0, lead(v=1.4, a=-1.0), DESIRED)     # above ~10 mph: official


def test_safe_launch_only_turns_constraint_off_and_latches_episode():
  sl = SafeLaunch()
  # Ordinary stop approach, no launch episode: official weighting kept.
  assert sl.prev_accel_constraint(True, 2.0, lead(v=1.0, a=-1.0, d=8.0), 9.0) is True
  # Standstill: stays official (False), launch gate opens an episode.
  assert sl.prev_accel_constraint(False, 0.0, lead(), DESIRED) is False and sl.episode
  # Moving in the launch with all gates: standstill weighting kept.
  assert sl.prev_accel_constraint(True, 0.5, lead(v=1.2, a=0.8, d=8.2), 7.9) is False
  # Lead brakes: gate fails, episode lets braking build without the change cost.
  assert sl.prev_accel_constraint(True, 1.0, lead(v=0.8, a=-1.5, d=8.0), 8.4) is False
  # Above ~10 mph the episode ends and official weighting returns.
  assert sl.prev_accel_constraint(True, 5.0, lead(v=6.0, a=0.0, d=20.0), 12.0) is True
  assert not sl.episode
  # Never turns an official False into True.
  assert sl.prev_accel_constraint(False, 3.0, lead(v=6.0, a=0.0, d=20.0), 12.0) is False


def test_release_prev_accel_is_brake_only():
  # Positive request at a standstill out of Stopping: deep hold -> light hold.
  assert release_prev_accel(-2.0, 0.12, 0.0, True, True) == LAUNCH_RELEASE_HOLD
  assert LAUNCH_RELEASE_HOLD < 0.0
  # Small positive request below the light hold is never exceeded.
  assert release_prev_accel(-2.0, 0.12, 0.0, True, True) <= 0.12
  # Braking / zero request, rolling, Stopping state, or inactive: unchanged.
  for args in [(-2.0, 0.0, 0.0, True, True), (-2.0, -1.0, 0.0, True, True), (-2.0, 0.3, 0.5, True, True),
               (-2.0, 0.3, 0.0, False, True), (-2.0, 0.3, 0.0, True, False), (0.1, 0.3, 0.0, True, True)]:
    assert release_prev_accel(*args) == args[0], args
