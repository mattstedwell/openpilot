"""Onroad screen must not wake automatically except for critical (red
take-control-immediately) alerts. Touch and the OnroadScreenOffTimer
inactivity timeout are unchanged. Sounds are not involved.

Loads UIStateSP / DeviceSP from source with stubs for gui_app and Params
so this runs on the host without pyray, msgq, or a display.
"""
import ast
from enum import Enum, IntEnum
from pathlib import Path
from types import SimpleNamespace

from openpilot.cereal import log

UI_STATE_PY = Path(__file__).resolve().parents[1] / "sunnypilot" / "ui_state.py"
CIRCULAR_PY = Path(__file__).resolve().parents[1] / "sunnypilot" / "onroad" / "circular_alerts.py"
BIG_VIEW = Path(__file__).resolve().parents[1] / "onroad" / "augmented_road_view.py"
MICI_VIEW = Path(__file__).resolve().parents[2] / "ui" / "mici" / "onroad" / "augmented_road_view.py"

FPS = 20
TIMEOUT_S = 30
FULL = TIMEOUT_S * FPS  # 600

AlertStatus = log.SelfdriveState.AlertStatus


class OnroadBrightness(IntEnum):
  AUTO = 0
  AUTO_DARK = 1
  SCREEN_OFF = 2


class _GuiApp:
  def __init__(self):
    self.target_fps = FPS
    self.mouse_events = []

  def clear(self):
    self.mouse_events.clear()


gui_app = _GuiApp()
# wake_from_dimmed calls gui_app.mouse_events.clear(); give the list a clear
# and also allow replacing the list with an object that has clear().
gui_app.mouse_events = []


def _load():
  tree = ast.parse(UI_STATE_PY.read_text())
  names = {"ONROAD_BRIGHTNESS_TIMER_PAUSED", "OnroadTimerStatus", "UIStateSP", "DeviceSP"}
  body = []
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
      body.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in names for t in node.targets):
      body.append(node)
  ns = {
    "Enum": Enum,
    "IntEnum": IntEnum,
    "gui_app": gui_app,
    "OnroadBrightness": OnroadBrightness,
    "log": log,
    "Params": object,
  }
  exec(compile(ast.Module(body=body, type_ignores=[]), str(UI_STATE_PY), "exec"), ns)
  return ns


_NS = _load()
UIStateSP = _NS["UIStateSP"]
DeviceSP = _NS["DeviceSP"]
OnroadTimerStatus = _NS["OnroadTimerStatus"]
ONROAD_BRIGHTNESS_TIMER_PAUSED = _NS["ONROAD_BRIGHTNESS_TIMER_PAUSED"]


def _host(timer=FULL, brightness=10):
  h = object.__new__(UIStateSP)
  h.onroad_brightness = brightness
  h.onroad_brightness_timer = timer
  h.onroad_brightness_timer_param = TIMEOUT_S
  h._onroad_timer_saved = None
  h.started = True
  return h


def _ui(started=True, started_frame=0, car_frame=1):
  sm = SimpleNamespace(recv_frame={"carState": car_frame})
  return SimpleNamespace(started=started, started_frame=started_frame, sm=sm)


def _alert(status, text1=""):
  return SimpleNamespace(status=status, text1=text1)


PERSONALITY = _alert(AlertStatus.normal, "Following Distance: Relaxed")
# raw int, as both alert renderers set status=ss.alertStatus.raw
PERSONALITY_RAW = _alert(int(getattr(AlertStatus.normal, "raw", AlertStatus.normal)),
                         "Following Distance: Relaxed")
PROMPT = _alert(AlertStatus.userPrompt, "Pay Attention")
CRITICAL = _alert(AlertStatus.critical, "TAKE CONTROL IMMEDIATELY")
CRITICAL_RAW = _alert(int(getattr(AlertStatus.critical, "raw", AlertStatus.critical)),
                      "TAKE CONTROL IMMEDIATELY")


def test_profile_change_alert_does_not_wake():
  """Personality/profile-change popup is a NormalPermanentAlert: must not
  restart the sleep timer (and must not freeze it either)."""
  for alert in (PERSONALITY, PERSONALITY_RAW, PROMPT):
    h = _host(timer=100)
    ui = _ui()
    h.onroad_brightness_handle_alerts(ui, alert)
    assert h.onroad_brightness_timer == 99, alert
    # still expired after running down; a fresh profile-change does not wake
    h.onroad_brightness_timer = 0
    h.onroad_brightness_handle_alerts(ui, alert)
    assert h.onroad_brightness_timer == 0
    assert h.onroad_brightness_timer_expired


def test_critical_alert_wakes():
  """Red take-control-immediately (alertStatus == critical) restarts the timer,
  including when the screen is already dimmed, and for both raw-int and enum
  status forms used by the mici and big alert renderers."""
  ui = _ui()
  for alert in (CRITICAL, CRITICAL_RAW):
    h = _host(timer=0)
    assert h.onroad_brightness_timer_expired
    h.onroad_brightness_handle_alerts(ui, alert)
    assert h.onroad_brightness_timer == FULL
    assert not h.onroad_brightness_timer_expired
    # mid-countdown: critical also restarts to full
    h.onroad_brightness_timer = 40
    h.onroad_brightness_handle_alerts(ui, alert)
    assert h.onroad_brightness_timer == FULL


def test_touch_wakes():
  """wake_from_dimmed_onroad_brightness (touch) is unchanged: a left-down
  while dimmed restarts the full timeout."""
  h = _host(timer=0)
  evs = [SimpleNamespace(left_down=True)]
  DeviceSP.wake_from_dimmed_onroad_brightness(h, evs)
  assert h.onroad_brightness_timer == FULL
  # no touch, already dimmed: stays dimmed
  h.onroad_brightness_timer = 0
  DeviceSP.wake_from_dimmed_onroad_brightness(h, [SimpleNamespace(left_down=False)])
  assert h.onroad_brightness_timer == 0
  # mid-countdown, not AUTO_DARK: touch does not retrigger (same as before)
  h.onroad_brightness_timer = 40
  DeviceSP.wake_from_dimmed_onroad_brightness(h, evs)
  assert h.onroad_brightness_timer == 40


def test_timer_counts_down_during_a_normal_alert():
  """Previously update_onroad_brightness returned early on has_alert, which
  froze the countdown for the whole popup. It must keep ticking."""
  h = _host(timer=10)
  ui = _ui()
  for i in range(10):
    h.onroad_brightness_handle_alerts(ui, PERSONALITY)
    assert h.onroad_brightness_timer == 9 - i
  assert h.onroad_brightness_timer == 0
  assert h.onroad_brightness_timer_expired
  # and stays expired through the rest of the popup
  h.onroad_brightness_handle_alerts(ui, PERSONALITY)
  assert h.onroad_brightness_timer == 0


def test_show_event_does_not_wake():
  """Returning to the driving view (show_event -> RESUME) must not restart
  the full timeout. If hide paused the timer, resume continues the remainder.
  If it wasn't paused, RESUME is a no-op."""
  # paused mid-countdown: restore remainder, not FULL
  h = _host(timer=40)
  h.reset_onroad_sleep_timer(OnroadTimerStatus.PAUSE)
  assert h.onroad_brightness_timer == ONROAD_BRIGHTNESS_TIMER_PAUSED
  # while paused, the tick must not run it to expired or change the saved value
  h.update_onroad_brightness()
  assert h.onroad_brightness_timer == ONROAD_BRIGHTNESS_TIMER_PAUSED
  h.reset_onroad_sleep_timer(OnroadTimerStatus.RESUME)
  assert h.onroad_brightness_timer == 40
  # and then it continues
  h.update_onroad_brightness()
  assert h.onroad_brightness_timer == 39

  # RESUME when not paused: leave the current countdown alone (do not wake)
  h = _host(timer=40)
  h.reset_onroad_sleep_timer(OnroadTimerStatus.RESUME)
  assert h.onroad_brightness_timer == 40

  # already expired, then hide and show: stay expired, do not wake
  h = _host(timer=0)
  h.reset_onroad_sleep_timer(OnroadTimerStatus.PAUSE)
  h.reset_onroad_sleep_timer(OnroadTimerStatus.RESUME)
  assert h.onroad_brightness_timer == 0
  assert h.onroad_brightness_timer_expired

  # both road views still wire hide/show to PAUSE/RESUME (behavior is in the timer)
  for path in (BIG_VIEW, MICI_VIEW):
    src = path.read_text()
    assert "ui_state.reset_onroad_sleep_timer(OnroadTimerStatus.RESUME)" in src, path
    assert "ui_state.reset_onroad_sleep_timer(OnroadTimerStatus.PAUSE)" in src, path


def test_green_light_and_lead_depart_do_not_reset_the_timer():
  src = CIRCULAR_PY.read_text()
  assert "reset_onroad_sleep_timer" not in src
  # visuals and the 3 s display timer are still there
  assert "GREEN\\nLIGHT" in src or "GREEN\nLIGHT" in src
  assert "LEAD VEHICLE" in src
  assert "self._e2e_alert_display_timer = 3 * gui_app.target_fps" in src


def test_param_init_and_touch_helpers_still_full_reset():
  """First param init and a bare reset() (the touch / init path) still write
  the full timeout, and do not do so when already paused."""
  src = UI_STATE_PY.read_text()
  assert "if not self._sp_initialized:" in src
  assert "self.reset_onroad_sleep_timer()" in src
  assert "def wake_from_dimmed_onroad_brightness" in src

  h = _host(timer=0)
  h.reset_onroad_sleep_timer()  # NONE: full reset
  assert h.onroad_brightness_timer == FULL

  # paused: NONE must not unpause or rewrite (same as before)
  h = _host(timer=40)
  h.reset_onroad_sleep_timer(OnroadTimerStatus.PAUSE)
  h.reset_onroad_sleep_timer()
  assert h.onroad_brightness_timer == ONROAD_BRIGHTNESS_TIMER_PAUSED
