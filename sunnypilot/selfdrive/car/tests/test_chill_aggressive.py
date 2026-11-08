import pytest

from cereal import log
from openpilot.common.constants import CV
from openpilot.sunnypilot.selfdrive.car.chill_aggressive import SpeedLimitCruiseTarget, is_chill_aggressive, ChillAggressiveLaunch


class TestSpeedLimitCruiseTarget:
  def setup_method(self):
    self.policy = SpeedLimitCruiseTarget()
    self.engage()

  def engage(self, experimental=False, personality=log.LongitudinalPersonality.aggressive, enabled=True, cancel=False):
    self.policy.update_state(enabled, cancel, experimental, personality)

  def target(self, limit_mph):
    return self.policy.target_kph(limit_mph * CV.MPH_TO_MS, 8, 145)

  def test_tracks_raw_speed_limit(self):
    assert self.target(45) == pytest.approx(50 * CV.MPH_TO_KPH)
    assert self.target(35) == pytest.approx(40 * CV.MPH_TO_KPH)
    assert not self.policy.overridden

  def test_manual_override_and_cancel_reengage(self):
    self.target(45)
    self.policy.manual_speed_changed()
    assert self.policy.overridden
    assert self.target(35) is None
    self.engage(cancel=True)
    assert self.target(35) is None
    self.engage()  # stale enabled message before cancel has propagated
    assert self.target(35) is None
    self.engage(enabled=False)
    assert not self.policy.overridden
    self.engage()
    assert self.target(35) == pytest.approx(40 * CV.MPH_TO_KPH)

  def test_disable_without_cancel_resets(self):
    self.target(45)
    self.policy.manual_speed_changed()
    self.engage(enabled=False)
    self.engage()
    assert self.target(35) == pytest.approx(40 * CV.MPH_TO_KPH)

  def test_profile_gating(self):
    for experimental in (False, True):
      for personality in log.LongitudinalPersonality.schema.enumerants.values():
        self.engage(experimental, personality)
        expected = not experimental and personality == log.LongitudinalPersonality.aggressive
        assert is_chill_aggressive(experimental, personality) == expected
        assert (self.target(45) is not None) == expected

  def test_mode_switch_preserves_override(self):
    self.target(45)
    self.engage(experimental=True)
    assert self.target(35) is None
    self.engage()
    assert self.target(35) == pytest.approx(40 * CV.MPH_TO_KPH)
    self.policy.manual_speed_changed()
    self.engage(personality=log.LongitudinalPersonality.standard)
    self.engage()
    assert self.target(45) is None

  def test_driver_change_outside_profile_still_latches(self):
    self.target(45)
    self.engage(experimental=True)
    self.policy.manual_speed_changed()
    self.engage()
    assert self.target(35) is None

  def test_change_before_first_application_does_not_lock(self):
    self.policy.manual_speed_changed()
    assert not self.policy.overridden
    assert self.target(45) is not None

  def test_invalid_limits_do_not_write_or_mark_applied(self):
    for limit in (0, -1, float('nan'), float('inf'), -float('inf')):
      assert self.target(limit) is None
      assert not self.policy.applied
    assert self.target(45) is not None
    assert self.target(0) is None
    assert not self.policy.overridden

  def test_cruise_bounds(self):
    assert self.policy.target_kph(.01, 20, 145) == 20
    assert self.target(100) == 145



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
