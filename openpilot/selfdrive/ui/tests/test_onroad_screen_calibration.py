"""While the camera is calibrating ("Calibration in progress"), the onroad
dark-screen / screen-off timer must not blank or dim the screen. Once
calibrated, the user's normal OnroadScreenOff behavior resumes after a full
countdown. Host-only: loads UIStateSP / DeviceSP from source with stubs.
"""
import ast
from enum import Enum, IntEnum
from pathlib import Path
from types import SimpleNamespace

from openpilot.cereal import log

UI_STATE_PY = Path(__file__).resolve().parents[1] / "sunnypilot" / "ui_state.py"
ROOT_UI_STATE_PY = Path(__file__).resolve().parents[1] / "ui_state.py"

FPS = 20
TIMEOUT_S = 3
FULL = TIMEOUT_S * FPS
Status = log.ExtrinsicsCalibration.Status


class OnroadBrightness(IntEnum):
  AUTO = 0
  AUTO_DARK = 1
  SCREEN_OFF = 2


gui_app = SimpleNamespace(target_fps=FPS, mouse_events=[])


def _load():
  tree = ast.parse(UI_STATE_PY.read_text())
  names = {"ONROAD_BRIGHTNESS_TIMER_PAUSED", "CALIBRATING_STATUSES", "OnroadTimerStatus", "UIStateSP", "DeviceSP"}
  body = []
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
      body.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
      body.append(node)
  ns = {"Enum": Enum, "IntEnum": IntEnum, "gui_app": gui_app, "OnroadBrightness": OnroadBrightness,
        "log": log, "Params": object}
  exec(compile(ast.Module(body=body, type_ignores=[]), str(UI_STATE_PY), "exec"), ns)
  return ns


_NS = _load()
UIStateSP = _NS["UIStateSP"]
DeviceSP = _NS["DeviceSP"]
OnroadTimerStatus = _NS["OnroadTimerStatus"]


class _SM:
  def __init__(self, status=None):
    self.status = status

  @property
  def seen(self):
    return {"extrinsicsCalibration": self.status is not None}

  def __getitem__(self, key):
    assert key == "extrinsicsCalibration"
    return SimpleNamespace(calStatus=self.status)


def _host(status, brightness=OnroadBrightness.SCREEN_OFF, timer=FULL, started=True):
  h = object.__new__(UIStateSP)
  h.onroad_brightness = brightness
  h.onroad_brightness_timer = timer
  h.onroad_brightness_timer_param = TIMEOUT_S
  h._onroad_timer_saved = None
  h.calibration_screen_hold = False
  h.started = started
  h.sm = _SM(status)
  return h


def _run(h, frames):
  """One UI frame: hold update (ui_state.update, before device.update), tick."""
  out = []
  for _ in range(frames):
    h.update_calibration_screen_hold()
    out.append(DeviceSP.set_onroad_brightness(h, True, 70.0))
    h.update_onroad_brightness()
  return out


def test_calibrating_statuses():
  for st, expected in ((Status.uncalibrated, True), (Status.recalibrating, True),
                       (Status.calibrated, False), (Status.invalid, False)):
    assert UIStateSP.calibration_in_progress(_host(st)) is expected
  assert not UIStateSP.calibration_in_progress(_host(None))
  assert not UIStateSP.calibration_in_progress(_host(Status.uncalibrated, started=False))


def test_screen_stays_on_while_calibrating():
  for brightness in (OnroadBrightness.SCREEN_OFF, OnroadBrightness.AUTO_DARK, 5):
    for st in (Status.uncalibrated, Status.recalibrating):
      h = _host(st, brightness=brightness)
      out = _run(h, FULL * 10)
      assert min(out) >= 30.0, (brightness, st)
      assert not h.onroad_brightness_timer_expired


def test_already_dark_wakes_when_recalibrating():
  h = _host(Status.recalibrating, timer=0)
  assert DeviceSP.set_onroad_brightness(h, True, 70.0) == 0.0  # before hold
  assert min(_run(h, 5)) == 70.0


def test_normal_dark_screen_resumes_after_calibration():
  h = _host(Status.uncalibrated)
  _run(h, FULL * 5)
  h.sm.status = Status.calibrated
  out = _run(h, FULL + 5)
  # full countdown after calibration completes, then the user's Screen Off
  assert all(b == 70.0 for b in out[:FULL])
  assert out[-1] == 0.0
  assert h.onroad_brightness_timer_expired
  assert not h.calibration_screen_hold


def test_calibrated_drive_unchanged():
  h = _host(Status.calibrated)
  out = _run(h, FULL + 2)
  assert out[-1] == 0.0 and h.onroad_brightness_timer_expired


def test_settings_pause_while_calibrating_still_bright():
  h = _host(Status.uncalibrated)
  h.reset_onroad_sleep_timer(OnroadTimerStatus.PAUSE)
  assert min(_run(h, 10)) == 70.0


def test_hold_runs_before_device_update():
  src = ROOT_UI_STATE_PY.read_text()
  assert src.index("UIStateSP.update_calibration_screen_hold(self)") < src.index("    device.update()")
