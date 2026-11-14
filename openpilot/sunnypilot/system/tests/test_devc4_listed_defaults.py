"""devc4-test2 defaults: ALC 0.5 s + BSM delay, Screen Off 3 s. No SSH/ADB/Chevron.

Host-safe: no real Params / libparams_c.so, no device network, no SSH/ADB
daemon or key changes. Official compiled params_keys.h defaults stay official;
runtime apply is in manager_init via apply_listed_defaults for ALC and screen
only. Stored SshEnabled / AdbEnabled / ChevronInfo values are never reset.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from openpilot.sunnypilot.system.devc4_listed_defaults import (
  DEFAULT_OVERRIDES,
  MIGRATION_FLAG_NAME,
  apply_listed_defaults,
  default_migration_flag_path,
)
import openpilot.sunnypilot.system.devc4_listed_defaults as defaults_mod

REPO = Path(__file__).resolve().parents[3]
MANAGER = REPO / "system" / "manager" / "manager.py"
PARAMS_KEYS = REPO / "common" / "params_keys.h"
AUTO_LANE = REPO / "sunnypilot" / "selfdrive" / "controls" / "lib" / "auto_lane_change.py"
SSH_KEY_WIDGET = REPO / "selfdrive" / "ui" / "widgets" / "ssh_key.py"
DISPLAY = REPO / "selfdrive" / "ui" / "sunnypilot" / "layouts" / "settings" / "display.py"
PARAMS_MIGRATION = REPO / "sunnypilot" / "system" / "params_migration.py"
HW_PY = REPO / "common" / "hardware" / "hw.py"
HW_H = REPO / "common" / "hardware" / "hw.h"
MICI_HUD = REPO / "selfdrive" / "ui" / "mici" / "onroad" / "hud_renderer.py"
SP_HUD = REPO / "selfdrive" / "ui" / "sunnypilot" / "onroad" / "hud_renderer.py"
STOCK_HUD = REPO / "selfdrive" / "ui" / "onroad" / "hud_renderer.py"
MICI_ONBOARD = REPO / "selfdrive" / "ui" / "mici" / "layouts" / "onboarding.py"
TICI_ONBOARD = REPO / "selfdrive" / "ui" / "layouts" / "onboarding.py"
SELFDRIVED = REPO / "selfdrive" / "selfdrived" / "selfdrived.py"
HUD_SUN = REPO / "selfdrive" / "assets" / "icons_mici" / "hud_sun.png"
HELPER = REPO / "sunnypilot" / "system" / "devc4_listed_defaults.py"
POP_NAME = REPO / "sunnypilot" / "models" / "model_name.py"
POLICY = REPO / "selfdrive" / "monitoring" / "policy.py"
PROFILES = REPO / "selfdrive" / "controls" / "lib" / "c4_long_profiles.py"


def _paths_from_source():
  """Load the real Paths class from hw.py without the package import graph.

  Other host tests may replace openpilot.common.hardware.hw with a stub
  (the prior manager crash was exactly a missing Paths.params on the class).
  """
  import os
  import platform
  from pathlib import Path as PathlibPath
  glb = {
    "os": os,
    "platform": platform,
    "Path": PathlibPath,
    "PC": True,
    "DEFAULT_DOWNLOAD_CACHE_ROOT": "/tmp/comma_download_cache",
  }
  lines = []
  for line in HW_PY.read_text().splitlines(True):
    if line.startswith("import ") or line.startswith("from "):
      continue
    lines.append(line)
  exec("".join(lines), glb)
  paths = glb["Paths"]
  if not hasattr(paths, "params") or not callable(paths.params):
    raise AssertionError("Paths.params is missing from hw.py; this crashed manager on device")
  return paths


Paths = _paths_from_source()


class FakeParams:
  def __init__(self, initial=None):
    self._d = dict(initial or {})

  def get(self, key):
    return self._d.get(key)

  def put(self, key, value, block=True):
    self._d[key] = value


class TestDefaultsIdentity(unittest.TestCase):
  def test_override_values_are_alc_and_screen_only(self):
    self.assertEqual(DEFAULT_OVERRIDES["AutoLaneChangeTimer"], 2)
    self.assertEqual(DEFAULT_OVERRIDES["AutoLaneChangeBsmDelay"], True)
    self.assertEqual(DEFAULT_OVERRIDES["OnroadScreenOffBrightness"], 2)
    self.assertEqual(DEFAULT_OVERRIDES["OnroadScreenOffTimer"], 3)
    self.assertEqual(set(DEFAULT_OVERRIDES), {
      "AutoLaneChangeTimer", "AutoLaneChangeBsmDelay",
      "OnroadScreenOffBrightness", "OnroadScreenOffTimer",
      "RoadEdgeLaneChangeEnabled", "Blendful", "SmartCruiseControlVision",
    })
    self.assertIs(DEFAULT_OVERRIDES["RoadEdgeLaneChangeEnabled"], True)
    self.assertIs(DEFAULT_OVERRIDES["Blendful"], True)
    self.assertIs(DEFAULT_OVERRIDES["SmartCruiseControlVision"], True)
    self.assertNotIn("SshEnabled", DEFAULT_OVERRIDES)
    self.assertNotIn("AdbEnabled", DEFAULT_OVERRIDES)
    self.assertNotIn("ChevronInfo", DEFAULT_OVERRIDES)
    self.assertEqual(MIGRATION_FLAG_NAME, "devc4_test2_listed_defaults_v1")
    self.assertNotEqual(MIGRATION_FLAG_NAME, "devc4_listed_defaults_v1")
    alc = AUTO_LANE.read_text()
    self.assertIn("NUDGE = 0", alc)
    self.assertIn("HALF_SECOND = 2", alc)
    self.assertIn("AutoLaneChangeMode.HALF_SECOND: 0.5", alc)
    display = DISPLAY.read_text()
    self.assertIn("SCREEN_OFF = 2", display)
    migration = PARAMS_MIGRATION.read_text()
    self.assertIn("{0: 3, 1: 5, 2: 7, 3: 10, 4: 15", migration)

  def test_source_params_keys_stay_official_and_manager_hooks(self):
    keys = PARAMS_KEYS.read_text()
    self.assertIn('{"SshEnabled", {PERSISTENT | BACKUP, BOOL}}', keys)
    self.assertIn('{"AdbEnabled", {PERSISTENT | BACKUP, BOOL}}', keys)
    self.assertIn('{"AutoLaneChangeTimer", {PERSISTENT | BACKUP, INT, "0"}}', keys)
    self.assertIn('{"AutoLaneChangeBsmDelay", {PERSISTENT | BACKUP, BOOL, "0"}}', keys)
    self.assertIn('{"ChevronInfo", {PERSISTENT | BACKUP, INT, "4"}}', keys)
    self.assertIn('{"OnroadScreenOffBrightness", {PERSISTENT | BACKUP, INT, "0"}}', keys)
    self.assertIn('{"OnroadScreenOffTimer", {PERSISTENT | BACKUP, INT, "15"}}', keys)
    self.assertNotIn("UnauthenticatedSsh", keys)
    self.assertIn("GithubSshKeys", keys)
    mgr = MANAGER.read_text()
    self.assertIn("apply_listed_defaults(params)", mgr)
    self.assertLess(mgr.index("params.get_default_value(k)"), mgr.index("apply_listed_defaults(params)"))
    helper = HELPER.read_text()
    self.assertNotIn("SshEnabled", helper.split("DEFAULT_OVERRIDES")[1].split("}")[0])
    self.assertNotIn("ChevronInfo", DEFAULT_OVERRIDES)
    self.assertNotIn('put("SshEnabled"', helper)
    self.assertNotIn('put("AdbEnabled"', helper)
    self.assertNotIn('put("ChevronInfo"', helper)
    self.assertNotIn("GithubSshKeys", helper)
    self.assertIn('MIGRATION_FLAG_NAME = "devc4_test2_listed_defaults_v1"', helper)
    self.assertNotIn('MIGRATION_FLAG_NAME = "devc4_listed_defaults_v1"', helper)


class TestApplyDefaults(unittest.TestCase):
  def _flag(self):
    return os.path.join(tempfile.mkdtemp(), "devc4_test2_listed_defaults_v1")

  def test_missing_params_get_alc_and_screen_only(self):
    params = FakeParams()
    flag = self._flag()
    apply_listed_defaults(params, flag_path=flag)
    self.assertEqual(params.get("AutoLaneChangeTimer"), 2)
    self.assertIs(params.get("AutoLaneChangeBsmDelay"), True)
    self.assertEqual(params.get("OnroadScreenOffBrightness"), 2)
    self.assertEqual(params.get("OnroadScreenOffTimer"), 3)
    self.assertIsNone(params.get("SshEnabled"))
    self.assertIsNone(params.get("AdbEnabled"))
    self.assertIsNone(params.get("ChevronInfo"))
    self.assertTrue(os.path.exists(flag))

  def test_official_alc_and_screen_are_migrated_once(self):
    params = FakeParams({
      "AutoLaneChangeTimer": 0,
      "AutoLaneChangeBsmDelay": False,
      "OnroadScreenOffBrightness": 0,
      "OnroadScreenOffTimer": 15,
    })
    flag = self._flag()
    self.assertTrue(apply_listed_defaults(params, flag_path=flag))
    self.assertEqual(params.get("AutoLaneChangeTimer"), 2)
    self.assertIs(params.get("AutoLaneChangeBsmDelay"), True)
    self.assertEqual(params.get("OnroadScreenOffBrightness"), 2)
    self.assertEqual(params.get("OnroadScreenOffTimer"), 3)
    self.assertTrue(os.path.exists(flag))
    params2 = FakeParams({
      "AutoLaneChangeTimer": 0,
      "AutoLaneChangeBsmDelay": False,
      "OnroadScreenOffBrightness": 0,
      "OnroadScreenOffTimer": 15,
    })
    self.assertFalse(apply_listed_defaults(params2, flag_path=flag))
    self.assertEqual(params2.get("AutoLaneChangeTimer"), 0)
    self.assertIs(params2.get("AutoLaneChangeBsmDelay"), False)
    self.assertEqual(params2.get("OnroadScreenOffTimer"), 15)

  def test_previous_unlisted_5s_screen_off_is_migrated_once(self):
    params = FakeParams({
      "OnroadScreenOffBrightness": 2,
      "OnroadScreenOffTimer": 5,
      "AutoLaneChangeTimer": 2,
      "AutoLaneChangeBsmDelay": True,
    })
    flag = self._flag()
    self.assertTrue(apply_listed_defaults(params, flag_path=flag))
    self.assertEqual(params.get("OnroadScreenOffBrightness"), 2)
    self.assertEqual(params.get("OnroadScreenOffTimer"), 3)
    params2 = FakeParams({
      "OnroadScreenOffBrightness": 2,
      "OnroadScreenOffTimer": 5,
    })
    self.assertFalse(apply_listed_defaults(params2, flag_path=flag))
    self.assertEqual(params2.get("OnroadScreenOffTimer"), 5)

  def test_user_overrides_preserved_on_first_and_later_apply(self):
    params = FakeParams({
      "AutoLaneChangeTimer": 3,
      "AutoLaneChangeBsmDelay": False,
      "OnroadScreenOffBrightness": 1,
      "OnroadScreenOffTimer": 10,
    })
    flag = self._flag()
    apply_listed_defaults(params, flag_path=flag)
    self.assertEqual(params.get("AutoLaneChangeTimer"), 3)
    self.assertIs(params.get("AutoLaneChangeBsmDelay"), False)
    self.assertEqual(params.get("OnroadScreenOffBrightness"), 1)
    self.assertEqual(params.get("OnroadScreenOffTimer"), 10)

  def test_does_not_fill_or_reset_ssh_adb_or_chevron(self):
    # Official unset SSH/ADB stay unset; official Chevron All stays All.
    params = FakeParams({"ChevronInfo": 4})
    flag = self._flag()
    apply_listed_defaults(params, flag_path=flag)
    self.assertIsNone(params.get("SshEnabled"))
    self.assertIsNone(params.get("AdbEnabled"))
    self.assertEqual(params.get("ChevronInfo"), 4)

    # Stored values from a previous build, or an explicit off, are not reset.
    params2 = FakeParams({
      "SshEnabled": True,
      "AdbEnabled": True,
      "ChevronInfo": 0,
      "AutoLaneChangeTimer": 2,
      "AutoLaneChangeBsmDelay": True,
      "OnroadScreenOffBrightness": 2,
      "OnroadScreenOffTimer": 3,
    })
    flag2 = self._flag()
    apply_listed_defaults(params2, flag_path=flag2)
    self.assertIs(params2.get("SshEnabled"), True)
    self.assertIs(params2.get("AdbEnabled"), True)
    self.assertEqual(params2.get("ChevronInfo"), 0)

    params3 = FakeParams({"SshEnabled": False, "AdbEnabled": False, "ChevronInfo": 1})
    apply_listed_defaults(params3, flag_path=self._flag())
    self.assertIs(params3.get("SshEnabled"), False)
    self.assertIs(params3.get("AdbEnabled"), False)
    self.assertEqual(params3.get("ChevronInfo"), 1)

  def test_prior_devc4_flag_does_not_skip_alc_or_screen(self):
    params = FakeParams({
      "AutoLaneChangeTimer": 0,
      "AutoLaneChangeBsmDelay": False,
      "OnroadScreenOffBrightness": 0,
      "OnroadScreenOffTimer": 15,
    })
    d = tempfile.mkdtemp()
    # An already-run devc4 flag must not prevent this branch's ALC/screen apply.
    Path(d, "devc4_listed_defaults_v1").write_text("1\n")
    flag = os.path.join(d, MIGRATION_FLAG_NAME)
    self.assertTrue(apply_listed_defaults(params, flag_path=flag))
    self.assertEqual(params.get("AutoLaneChangeTimer"), 2)
    self.assertEqual(params.get("OnroadScreenOffTimer"), 3)
    self.assertTrue(os.path.isfile(flag))

  def test_screensaver_not_rewritten(self):
    params = FakeParams({"ScreenSaverEnabled": True, "OnroadScreenOffBrightness": 0, "OnroadScreenOffTimer": 15})
    flag = self._flag()
    apply_listed_defaults(params, flag_path=flag)
    self.assertIs(params.get("ScreenSaverEnabled"), True)
    self.assertEqual(params.get("OnroadScreenOffTimer"), 3)


class TestNoKeyInstall(unittest.TestCase):
  def test_ssh_key_mechanisms_untouched(self):
    src = HELPER.read_text()
    self.assertIn("GithubSshKeys", SSH_KEY_WIDGET.read_text())
    self.assertNotIn("GithubSshKeys", src)
    self.assertNotIn("authorized_keys", src)
    self.assertNotIn("id_rsa", src)
    self.assertNotIn("ScreenSaverEnabled", DEFAULT_OVERRIDES)


class TestRemovedAndRetainedScope(unittest.TestCase):
  def test_official_hud_and_engagement_icon_restored(self):
    mici = MICI_HUD.read_text()
    self.assertIn("gui_app.texture('icons_mici/wheel.png', 50, 50)", mici)
    self.assertNotIn("hud_sun", mici)
    self.assertFalse(HUD_SUN.exists())
    # MAX / set-speed is drawn again; current speed remains official (defined, not called)
    self.assertIn("self._draw_set_speed(rect)", mici)
    self.assertNotIn("listed: hide", mici.lower() + SP_HUD.read_text().lower() + STOCK_HUD.read_text().lower())
    sp = SP_HUD.read_text()
    self.assertIn("self.speed_limit_renderer.render(rect)", sp)
    self.assertIn("self.speed_renderer.render(rect)", sp)
    stock = STOCK_HUD.read_text()
    self.assertIn("self._draw_set_speed(rect)", stock)
    self.assertIn("self._draw_current_speed(rect)", stock)

  def test_tutorial_uses_official_version_logic(self):
    for path in (MICI_ONBOARD, TICI_ONBOARD):
      src = path.read_text()
      self.assertIn('self._training_done: bool = ui_state.params.get("CompletedTrainingVersion") == training_version', src)
      self.assertNotIn("Listed: skip the training tutorial", src)
      # Official completion still writes the version only in the training callback.
      self.assertIn("def _on_completed_training", src)
      self.assertEqual(src.count('params.put("CompletedTrainingVersion"'), 1)
      self.assertIn('params.put("CompletedTrainingVersion", training_version, block=True)', src)

  def test_startup_master_suppression_removed_upstream_mici_hide_remains(self):
    src = SELFDRIVED.read_text()
    self.assertIn("if HARDWARE.get_device_type() == 'mici':", src)
    self.assertIn("self.startup_event = None", src)
    self.assertNotIn("Listed: never show the untested/master startup banner", src)
    self.assertNotIn("if self.startup_event == EventName.startupMaster:", src)
    # Other startup events are still assigned
    self.assertIn("EventName.startupNoCar", src)
    self.assertIn("EventName.startupNoControl", src)
    self.assertIn("EventName.startupNoSecOcKey", src)

  def test_driving_left_stock_and_dm_still_present(self):
    # This branch does not port longitudinal/launch/follow/accel/braking/turn
    # changes from devc4-test. Those helper files must not appear here.
    self.assertFalse(PROFILES.exists(), "c4_long_profiles.py must not be ported onto release-mici")
    model = POP_NAME.read_text()
    self.assertIn('DEFAULT_MODEL = "CD210"', model)
    self.assertNotIn("Pop Model (March 20, 2026)", model)
    self.assertIn("DM_ENFORCEMENT_DISABLED = True", POLICY.read_text())


class TestMigrationFlagPath(unittest.TestCase):
  def setUp(self):
    self._orig_helper_paths = defaults_mod.Paths
    defaults_mod.Paths = Paths

  def tearDown(self):
    defaults_mod.Paths = self._orig_helper_paths

  def test_paths_params_exists_on_actual_class_surface(self):
    self.assertTrue(hasattr(Paths, "params"), "Paths.params is missing; this crashed manager on device")
    self.assertTrue(callable(Paths.params))
    surface = {name for name, value in vars(Paths).items() if callable(value)}
    self.assertIn("params", surface)
    self.assertIn("comma_home", surface)
    self.assertIn("persist_root", surface)
    self.assertNotIn("get_param_path", surface)
    self.assertIn("def params()", HW_PY.read_text())
    self.assertIn("Must match C++ Path::params()", HW_PY.read_text())
    self.assertIn("inline std::string params()", HW_H.read_text())
    self.assertIn('util::getenv("PARAMS_ROOT"', HW_H.read_text())

  def test_params_root_env_and_openpilot_prefix(self):
    saved = {k: os.environ.get(k) for k in ("PARAMS_ROOT", "OPENPILOT_PREFIX")}
    try:
      os.environ.pop("PARAMS_ROOT", None)
      os.environ["OPENPILOT_PREFIX"] = "_devc4test2"
      expected = os.path.join(Paths.comma_home(), "params")
      self.assertTrue(expected.endswith(os.path.join(".comma_devc4test2", "params")))
      self.assertEqual(Paths.params(), expected)
      self.assertEqual(default_migration_flag_path(), os.path.join(expected, MIGRATION_FLAG_NAME))

      with tempfile.TemporaryDirectory() as d:
        os.environ["PARAMS_ROOT"] = d
        self.assertEqual(Paths.params(), d)
        flag = default_migration_flag_path()
        self.assertEqual(flag, os.path.join(d, MIGRATION_FLAG_NAME))
        self.assertEqual(os.path.dirname(flag), d)
        self.assertNotEqual(os.path.basename(flag), "d")
        self.assertFalse(flag.startswith(os.path.join(d, "d") + os.sep))
    finally:
      for k, v in saved.items():
        if v is None:
          os.environ.pop(k, None)
        else:
          os.environ[k] = v

  def test_apply_uses_real_paths_params_without_injected_flag(self):
    saved = os.environ.get("PARAMS_ROOT")
    with tempfile.TemporaryDirectory() as d:
      os.environ["PARAMS_ROOT"] = d
      try:
        params = FakeParams()
        apply_listed_defaults(params)
        flag = os.path.join(d, MIGRATION_FLAG_NAME)
        self.assertTrue(os.path.isfile(flag))
        self.assertEqual(params.get("AutoLaneChangeTimer"), 2)
        self.assertEqual(params.get("OnroadScreenOffTimer"), 3)
        self.assertIsNone(params.get("SshEnabled"))
        self.assertFalse(os.path.exists(os.path.join(d, "devc4_listed_defaults_v1")))
      finally:
        if saved is None:
          os.environ.pop("PARAMS_ROOT", None)
        else:
          os.environ["PARAMS_ROOT"] = saved

  def test_stat_oserror_does_not_crash_or_rewrite_or_mark(self):
    params = FakeParams({"AutoLaneChangeTimer": 0, "AutoLaneChangeBsmDelay": False})
    flag = os.path.join(tempfile.mkdtemp(), "devc4_test2_listed_defaults_v1")
    real_exists = os.path.exists

    def boom(path):
      if path == flag:
        raise OSError("stat failed")
      return real_exists(path)

    orig = os.path.exists
    os.path.exists = boom
    try:
      apply_listed_defaults(params, flag_path=flag)
    finally:
      os.path.exists = orig
    self.assertEqual(params.get("AutoLaneChangeTimer"), 0)
    self.assertIs(params.get("AutoLaneChangeBsmDelay"), False)
    self.assertFalse(os.path.exists(flag))

  def test_write_oserror_does_not_crash_or_mark_complete(self):
    with tempfile.TemporaryDirectory() as d:
      blocker = os.path.join(d, "notdir")
      with open(blocker, "w") as f:
        f.write("x")
      flag = os.path.join(blocker, "flag")
      params = FakeParams()
      apply_listed_defaults(params, flag_path=flag)
      self.assertEqual(params.get("AutoLaneChangeTimer"), 2)
      self.assertFalse(os.path.exists(flag))
      params_after = FakeParams({
        "AutoLaneChangeTimer": 2,
        "AutoLaneChangeBsmDelay": True,
        "OnroadScreenOffBrightness": 2,
        "OnroadScreenOffTimer": 3,
      })
      writable = os.path.join(d, MIGRATION_FLAG_NAME)
      apply_listed_defaults(params_after, flag_path=writable)
      self.assertTrue(os.path.isfile(writable))
      self.assertEqual(params_after.get("AutoLaneChangeTimer"), 2)
      self.assertEqual(params_after.get("OnroadScreenOffTimer"), 3)


if __name__ == "__main__":
  unittest.main()


class TestFillMissingToggles(unittest.TestCase):
  class _P:
    def __init__(self, d): self.d = dict(d)
    def get(self, k): return self.d.get(k)
    def put(self, k, v, block=True): self.d[k] = v

  def test_missing_toggles_filled_and_stored_values_kept(self):
    import tempfile, os
    from openpilot.sunnypilot.system.devc4_listed_defaults import apply_listed_defaults
    with tempfile.TemporaryDirectory() as d:
      p = self._P({"Blendful": False})
      apply_listed_defaults(p, flag_path=os.path.join(d, "flag"))
      self.assertIs(p.d["RoadEdgeLaneChangeEnabled"], True)
      self.assertIs(p.d["SmartCruiseControlVision"], True)
      self.assertIs(p.d["Blendful"], False)  # user-stored value left alone


class TestToggleDefaultsOnce(unittest.TestCase):
  """dev-c4-test (ported from dev-c4-sunnypilot-lite 61db6a002d): RoadEdge / Blendful / SCC Vision reach the device.

  Simulates manager_init order on the device: the official loop stores the
  prebuilt libparams_c.so compiled default (False) first, then
  apply_listed_defaults, then apply_toggle_defaults.
  """
  KEYS = ("RoadEdgeLaneChangeEnabled", "Blendful", "SmartCruiseControlVision")

  class _P:
    def __init__(self, d): self.d = dict(d)
    def get(self, k): return self.d.get(k)
    def put(self, k, v, block=True): self.d[k] = v

  def _boot(self, p, d):
    from openpilot.sunnypilot.system.devc4_listed_defaults import apply_toggle_defaults
    for k in self.KEYS:  # official "set unset params to default" with prebuilt "0"
      if p.get(k) is None:
        p.put(k, False)
    apply_listed_defaults(p, flag_path=os.path.join(d, "listed"))
    return apply_toggle_defaults(p, flag_path=os.path.join(d, "toggles"))

  def test_fresh_device_gets_toggles_on_despite_prebuilt_off_default(self):
    with tempfile.TemporaryDirectory() as d:
      p = self._P({})
      self.assertTrue(self._boot(p, d))
      for k in self.KEYS:
        self.assertIs(p.d[k], True, k)

  def test_existing_device_with_official_off_is_turned_on_once(self):
    with tempfile.TemporaryDirectory() as d:
      p = self._P({k: False for k in self.KEYS})
      self._boot(p, d)
      for k in self.KEYS:
        self.assertIs(p.d[k], True, k)

  def test_later_user_off_is_respected(self):
    with tempfile.TemporaryDirectory() as d:
      p = self._P({})
      self._boot(p, d)
      p.put("Blendful", False)  # user turns it off in Settings
      self.assertFalse(self._boot(p, d))
      self.assertIs(p.d["Blendful"], False)
      self.assertIs(p.d["RoadEdgeLaneChangeEnabled"], True)

  def test_flag_failure_does_not_write(self):
    from openpilot.sunnypilot.system.devc4_listed_defaults import apply_toggle_defaults
    orig = defaults_mod._flag_already_present
    defaults_mod._flag_already_present = lambda _p: None
    try:
      p = self._P({k: False for k in self.KEYS})
      self.assertFalse(apply_toggle_defaults(p, flag_path="/nonexistent/x"))
      for k in self.KEYS:
        self.assertIs(p.d[k], False)
    finally:
      defaults_mod._flag_already_present = orig

  def test_manager_calls_toggle_defaults_after_official_loop(self):
    src = MANAGER.read_text()
    loop = src.index("set unset params to their default value")
    self.assertGreater(src.index("apply_toggle_defaults(params)"), loop)
    self.assertGreater(src.index("apply_listed_defaults(params)"), loop)


  def test_own_flag_name(self):
    self.assertEqual(defaults_mod.TOGGLE_FLAG_NAME, "devc4_test_toggle_defaults_v1")
    self.assertNotEqual(defaults_mod.TOGGLE_FLAG_NAME, defaults_mod.MIGRATION_FLAG_NAME)
