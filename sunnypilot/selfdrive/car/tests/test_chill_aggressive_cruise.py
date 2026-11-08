"""Cruise-helper integration tests; use the normal native openpilot test environment."""
import pytest

from cereal import car, custom, log
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import VCruiseHelper, V_CRUISE_MAX
from openpilot.sunnypilot.selfdrive.car import cruise_ext

ButtonType = car.CarState.ButtonEvent.Type


class CruiseParams:
  def get_bool(self, key):
    return False

  def get(self, key, **kwargs):
    return 1


@pytest.fixture(params=[(False, True), (False, False), (True, True), (True, False)])
def cruise(request, monkeypatch):
  pcm, pcm_speed = request.param
  monkeypatch.setattr(cruise_ext, 'Params', CruiseParams)
  helper = VCruiseHelper(car.CarParams(pcmCruise=pcm, openpilotLongitudinalControl=True),
                         custom.CarParamsSP(pcmCruiseSpeed=pcm_speed))
  helper.enable_button_timers = dict(cruise_ext.CRUISE_BUTTON_TIMER)
  return helper


def step(helper, *, enabled=True, limit=45, button=None, pressed=False, pcm_speed=45,
         experimental=False, personality=log.LongitudinalPersonality.aggressive, standstill=False):
  cs = car.CarState(vEgo=30 * CV.MPH_TO_MS,
                    cruiseState={'available': True, 'enabled': enabled, 'standstill': standstill,
                                 'speed': pcm_speed * CV.MPH_TO_MS, 'speedCluster': pcm_speed * CV.MPH_TO_MS})
  if button is not None:
    cs.buttonEvents = [car.CarState.ButtonEvent(type=button, pressed=pressed)]
  helper.detected_speed_limit = limit * CV.MPH_TO_MS
  helper.update_v_cruise(cs, enabled, False, experimental, personality)
  return cs


def engage(helper, limit=45):
  step(helper, enabled=False, limit=limit)
  cs = step(helper, limit=limit)
  helper.initialize_v_cruise(cs, False, False)  # card's actual engage ordering
  step(helper, limit=limit)


def test_engage_and_limit_changes(cruise):
  engage(cruise)
  assert cruise.v_cruise_kph == pytest.approx(50 * CV.MPH_TO_KPH)
  step(cruise, limit=35)
  assert cruise.v_cruise_kph == pytest.approx(40 * CV.MPH_TO_KPH)
  assert not cruise.speed_limit_target.overridden


def test_buttons_override_until_cancel_and_reengage(cruise):
  engage(cruise, limit=35)
  # PCM-owned speed uses the PCM-reported driver delta; software cruise uses
  # the normal button press/release handler. In both cases, two '-' taps stick.
  for stock_speed in (44, 43):
    step(cruise, limit=35, button=ButtonType.decelCruise, pressed=True, pcm_speed=stock_speed + 1)
    step(cruise, limit=35, button=ButtonType.decelCruise, pressed=False, pcm_speed=stock_speed)
  assert cruise.v_cruise_kph * CV.KPH_TO_MPH == pytest.approx(38, abs=.1)
  target = cruise.v_cruise_kph
  assert cruise.speed_limit_target.overridden
  step(cruise, limit=55, pcm_speed=43)
  assert cruise.v_cruise_kph == target
  step(cruise, limit=35, button=ButtonType.cancel, pressed=True, pcm_speed=43)
  assert not cruise.speed_limit_target.enabled
  engage(cruise, limit=35)
  assert cruise.v_cruise_kph == pytest.approx(40 * CV.MPH_TO_KPH)
  assert not cruise.speed_limit_target.overridden


def test_no_limit_and_mode_exit_do_not_yank_target(cruise):
  engage(cruise)
  expected = cruise.v_cruise_kph
  step(cruise, limit=0)
  assert cruise.v_cruise_kph == expected
  step(cruise, limit=35, experimental=True)
  assert cruise.v_cruise_kph == expected
  step(cruise, limit=35, personality=log.LongitudinalPersonality.relaxed)
  assert cruise.v_cruise_kph == expected
  step(cruise, limit=35)
  assert cruise.v_cruise_kph == pytest.approx(40 * CV.MPH_TO_KPH)


def test_does_not_stack_legacy_sla_offsets(cruise):
  cruise.sla_state = custom.LongitudinalPlanSP.SpeedLimit.AssistState.active
  cruise.speed_limit_final_last_kph = 100.
  cruise.has_speed_limit = True
  engage(cruise)
  assert cruise.v_cruise_kph == pytest.approx(50 * CV.MPH_TO_KPH)
  step(cruise, limit=100)
  assert cruise.v_cruise_kph == V_CRUISE_MAX


def test_enable_button_release_is_not_override(cruise):
  step(cruise, enabled=False, button=ButtonType.decelCruise, pressed=True)
  cs = step(cruise, button=ButtonType.decelCruise, pressed=False)
  cruise.initialize_v_cruise(cs, False, False)
  step(cruise)
  assert not cruise.speed_limit_target.overridden
  assert cruise.v_cruise_kph == pytest.approx(50 * CV.MPH_TO_KPH)
