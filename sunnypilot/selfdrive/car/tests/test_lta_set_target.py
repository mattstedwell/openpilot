import pytest

from cereal import car
from openpilot.common.constants import CV
from openpilot.sunnypilot.selfdrive.car.lta_set_target import LtaSetTarget

ButtonType = car.CarState.ButtonEvent.Type


def press(btn):
  return [car.CarState.ButtonEvent(type=btn, pressed=True), car.CarState.ButtonEvent(type=btn, pressed=False)]


class TestLtaSetTarget:
  def setup_method(self):
    self.policy = LtaSetTarget()

  def step(self, buttons=(), limit_mph=45., available=True, enabled=True):
    return self.policy.update(list(buttons), available, enabled, limit_mph * CV.MPH_TO_MS, 8, 145)

  def test_off_until_lta_press(self):
    assert self.step() is None

  def test_press_sets_limit_plus_5_and_tracks_limit(self):
    assert self.step(press(ButtonType.setCruise)) == pytest.approx(50 * CV.MPH_TO_KPH)
    assert self.step(limit_mph=35) == pytest.approx(40 * CV.MPH_TO_KPH)

  def test_no_limit_does_not_invent_speed(self):
    assert self.step(press(ButtonType.setCruise), limit_mph=0) is None
    assert not self.policy.active

  def test_lost_limit_holds_last_target(self):
    self.step(press(ButtonType.setCruise))
    assert self.step(limit_mph=0) == pytest.approx(50 * CV.MPH_TO_KPH)
    assert self.step(limit_mph=float('nan')) == pytest.approx(50 * CV.MPH_TO_KPH)

  def test_cancel_turns_off_until_next_press(self):
    self.step(press(ButtonType.setCruise))
    assert self.step(press(ButtonType.cancel)) is None
    assert self.step() is None
    assert self.step(press(ButtonType.setCruise)) is not None

  def test_pcm_disengage_turns_off(self):
    self.step(press(ButtonType.setCruise))
    assert self.step(enabled=False) is None
    assert self.step(enabled=True) is None  # no re-apply on stalk re-engage without a new LTA press

  def test_main_off_turns_off(self):
    self.step(press(ButtonType.setCruise))
    assert self.step(available=False) is None
    assert self.step() is None

  def test_press_while_pcm_off_arms_target(self):
    assert self.step(press(ButtonType.setCruise), enabled=False) == pytest.approx(50 * CV.MPH_TO_KPH)
    assert self.step(enabled=True) == pytest.approx(50 * CV.MPH_TO_KPH)

  def test_bounds(self):
    assert self.step(press(ButtonType.setCruise), limit_mph=100) == 145
