from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install_host_stubs
_install_host_stubs()

import pytest

from openpilot.cereal import log
from openpilot.sunnypilot.selfdrive.car.chill_aggressive import is_chill_aggressive, ChillAggressiveLaunch


class TestIsChillAggressive:
  def test_profile_gating(self):
    for experimental in (False, True):
      for personality in log.LongitudinalPersonality.schema.enumerants.values():
        expected = not experimental and personality == log.LongitudinalPersonality.aggressive
        assert is_chill_aggressive(experimental, personality) == expected


class TestChillAggressiveLaunch:
  def setup_method(self):
    self.launch = ChillAggressiveLaunch()

  def update(self, speed=0., target=20., enabled=True, experimental=False,
             personality=log.LongitudinalPersonality.aggressive, brake=False):
    return self.launch.update(enabled, experimental, personality, speed, target, brake)

  def test_only_arms_at_a_stop(self):
    assert not self.update(speed=3.)
    assert self.update()
    assert self.update(speed=3.)
    assert self.update(speed=9.)
    assert not self.update(speed=10.)
    assert not self.update(speed=3.)

  def test_stops_at_target(self):
    assert self.update(target=5.)
    assert self.update(speed=3., target=5.)
    assert not self.update(speed=4.5, target=5.)

  @pytest.mark.parametrize('experimental,personality', [
    (True, log.LongitudinalPersonality.aggressive),
    (False, log.LongitudinalPersonality.standard),
    (False, log.LongitudinalPersonality.relaxed),
  ])
  def test_other_profiles_never_arm(self, experimental, personality):
    assert not self.update(experimental=experimental, personality=personality)
    assert self.update()
    assert not self.update(speed=3., experimental=experimental, personality=personality)

  def test_disable_and_driver_brake_reset(self):
    assert self.update()
    assert not self.update(enabled=False)
    assert not self.update(speed=3.)
    assert self.update()
    assert not self.update(brake=True)
    assert not self.update(speed=3.)

  def test_invalid_target_does_not_launch(self):
    for target in (0., -1., float('nan'), float('inf')):
      assert not self.update(target=target)

  def test_launch_budget_tapers_to_stock(self):
    assert self.launch.accel_extra(0.) == pytest.approx(.4)
    assert self.launch.accel_extra(5.) == pytest.approx(.2)
    for speed in (10., 20., 40.):
      assert self.launch.accel_extra(speed) == 0.
