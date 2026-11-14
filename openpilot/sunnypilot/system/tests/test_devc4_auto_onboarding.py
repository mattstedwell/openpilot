"""devc4-test2: terms auto-accepted, sunnylink enabled once, "what is sunnypilot?"
notice skipped. Host-safe: fake Params (prebuilt libparams_c.so is aarch64)."""
from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path

from openpilot.common.version import (
  terms_version, terms_version_sp, training_version, sunnylink_consent_version, sunnylink_consent_declined,
)
import openpilot.sunnypilot.system.devc4_auto_onboarding as mod
from openpilot.sunnypilot.system.devc4_auto_onboarding import apply_auto_onboarding, SUNNYLINK_FLAG_NAME

REPO = Path(__file__).resolve().parents[3]
MANAGER = REPO / "system" / "manager" / "manager.py"
PARAMS_KEYS = REPO / "common" / "params_keys.h"
MICI_ONBOARD = REPO / "selfdrive" / "ui" / "mici" / "layouts" / "onboarding.py"
HARDWARED = REPO / "system" / "hardware" / "hardwared.py"
KEYS = ("HasAcceptedTerms", "HasAcceptedTermsSP", "SunnylinkEnabled", "CompletedSunnylinkConsentVersion",
        "CompletedTrainingVersion", "RecordFront", "RecordFrontLock")


class FakeParams:
  """Mimics openpilot.common.params.Params: STRING keys return str, BOOL via put_bool/get_bool."""
  def __init__(self, store=None, fail_keys=()):
    self.store = dict(store or {})
    self.fail_keys = set(fail_keys)
    self.writes = []

  def _check(self, key):
    if key not in KEYS:
      raise AssertionError(f"unexpected key {key}")
    if key in self.fail_keys:
      raise OSError(f"boom {key}")

  def get(self, key, block=False, return_default=False):
    self._check(key)
    return self.store.get(key)

  def get_bool(self, key, block=False):
    self._check(key)
    return bool(self.store.get(key))

  def put(self, key, dat, block=False):
    self._check(key)
    assert isinstance(dat, str), (key, dat)
    self.store[key] = dat
    self.writes.append(key)

  def put_bool(self, key, val, block=False):
    self._check(key)
    assert isinstance(val, bool)
    self.store[key] = val
    self.writes.append(key)


class TestAutoOnboarding(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.flag = os.path.join(self.tmp.name, SUNNYLINK_FLAG_NAME)

  def tearDown(self):
    self.tmp.cleanup()

  def test_fresh_device(self):
    p = FakeParams()
    apply_auto_onboarding(p, self.flag)
    self.assertEqual(p.store["HasAcceptedTerms"], terms_version)
    self.assertEqual(p.store["HasAcceptedTermsSP"], terms_version_sp)
    self.assertIs(p.store["SunnylinkEnabled"], True)
    self.assertEqual(p.store["CompletedSunnylinkConsentVersion"], sunnylink_consent_version)
    self.assertEqual(p.store["CompletedTrainingVersion"], training_version)  # DM intro + face check skipped
    self.assertIs(p.store["RecordFront"], False)  # driver camera data sharing declined
    self.assertTrue(os.path.exists(self.flag))

  def test_terms_version_bump_reaccepted_every_boot(self):
    p = FakeParams({"HasAcceptedTerms": "1", "HasAcceptedTermsSP": "0.9"})
    apply_auto_onboarding(p, self.flag)
    self.assertEqual(p.store["HasAcceptedTerms"], terms_version)
    self.assertEqual(p.store["HasAcceptedTermsSP"], terms_version_sp)
    # already current: no rewrite
    p.writes.clear()
    apply_auto_onboarding(p, self.flag)
    self.assertNotIn("HasAcceptedTerms", p.writes)
    self.assertNotIn("HasAcceptedTermsSP", p.writes)

  def test_sunnylink_enabled_once_then_user_off_respected(self):
    p = FakeParams({"SunnylinkEnabled": False, "CompletedSunnylinkConsentVersion": sunnylink_consent_declined})
    apply_auto_onboarding(p, self.flag)  # first run: enabled even if previously declined
    self.assertIs(p.store["SunnylinkEnabled"], True)
    self.assertEqual(p.store["CompletedSunnylinkConsentVersion"], sunnylink_consent_version)
    # user turns it off in Settings later
    p.store["SunnylinkEnabled"] = False
    for _ in range(3):
      apply_auto_onboarding(p, self.flag)
    self.assertIs(p.store["SunnylinkEnabled"], False)

  def test_consent_bump_never_blocks_but_keeps_enabled_state(self):
    open(self.flag, "w").close()  # one-time enable already done
    p = FakeParams({"SunnylinkEnabled": False, "CompletedSunnylinkConsentVersion": "0.5"})
    apply_auto_onboarding(p, self.flag)
    self.assertEqual(p.store["CompletedSunnylinkConsentVersion"], sunnylink_consent_version)
    self.assertIs(p.store["SunnylinkEnabled"], False)
    # an explicit decline (after the one-time enable) is kept
    p.store["CompletedSunnylinkConsentVersion"] = sunnylink_consent_declined
    apply_auto_onboarding(p, self.flag)
    self.assertEqual(p.store["CompletedSunnylinkConsentVersion"], sunnylink_consent_declined)

  def test_failures_never_raise_and_retry_next_boot(self):
    p = FakeParams(fail_keys={"SunnylinkEnabled", "HasAcceptedTerms"})
    apply_auto_onboarding(p, self.flag)  # must not raise
    self.assertEqual(p.store["HasAcceptedTermsSP"], terms_version_sp)
    self.assertFalse(os.path.exists(self.flag))  # not marked done -> retried
    p.fail_keys.clear()
    apply_auto_onboarding(p, self.flag)
    self.assertIs(p.store["SunnylinkEnabled"], True)
    self.assertTrue(os.path.exists(self.flag))

  def test_unwritable_flag_dir_does_not_raise(self):
    p = FakeParams()
    apply_auto_onboarding(p, "/proc/definitely/not/writable/flag")
    self.assertEqual(p.store["HasAcceptedTerms"], terms_version)
    self.assertIs(p.store["SunnylinkEnabled"], True)

  def test_totally_broken_params_does_not_raise(self):
    apply_auto_onboarding(object(), self.flag)

  def test_flag_lives_beside_params_store(self):
    self.assertEqual(SUNNYLINK_FLAG_NAME, "devc4_test2_sunnylink_enable_v1")
    src = Path(mod.__file__).read_text()
    self.assertIn("os.path.join(Paths.params(), SUNNYLINK_FLAG_NAME)", src)


class TestRecordFrontDecline(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.flag = os.path.join(self.tmp.name, SUNNYLINK_FLAG_NAME)

  def tearDown(self):
    self.tmp.cleanup()

  def test_unset_is_declined_once(self):
    p = FakeParams()
    apply_auto_onboarding(p, self.flag)
    self.assertIs(p.store["RecordFront"], False)
    p.writes.clear()
    apply_auto_onboarding(p, self.flag)
    self.assertNotIn("RecordFront", p.writes)

  def test_later_opt_in_respected(self):
    p = FakeParams()
    apply_auto_onboarding(p, self.flag)
    p.store["RecordFront"] = True  # user enables "record & upload cabin camera" in Settings
    for _ in range(3):
      apply_auto_onboarding(p, self.flag)
    self.assertIs(p.store["RecordFront"], True)

  def test_existing_choice_and_fleet_lock_untouched(self):
    p = FakeParams({"RecordFront": True})
    apply_auto_onboarding(p, self.flag)
    self.assertIs(p.store["RecordFront"], True)
    p = FakeParams({"RecordFrontLock": True})
    apply_auto_onboarding(p, self.flag)
    self.assertNotIn("RecordFront", p.store)

  def test_failure_does_not_raise_or_block_others(self):
    p = FakeParams(fail_keys={"RecordFront"})
    apply_auto_onboarding(p, self.flag)
    self.assertEqual(p.store["HasAcceptedTerms"], terms_version)
    self.assertIs(p.store["SunnylinkEnabled"], True)


class TestAutoCompleteTraining(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.flag = os.path.join(self.tmp.name, SUNNYLINK_FLAG_NAME)

  def tearDown(self):
    self.tmp.cleanup()

  def test_param_default_zero_is_replaced(self):
    p = FakeParams({"CompletedTrainingVersion": "0"})
    apply_auto_onboarding(p, self.flag)
    self.assertEqual(p.store["CompletedTrainingVersion"], training_version)

  def test_future_version_bump_recompleted(self):
    p = FakeParams({"CompletedTrainingVersion": "0.1.9"})
    apply_auto_onboarding(p, self.flag)
    self.assertEqual(p.store["CompletedTrainingVersion"], training_version)

  def test_current_value_not_rewritten(self):
    p = FakeParams({"CompletedTrainingVersion": training_version})
    apply_auto_onboarding(p, self.flag)
    self.assertNotIn("CompletedTrainingVersion", p.writes)
    apply_auto_onboarding(p, self.flag)
    self.assertNotIn("CompletedTrainingVersion", p.writes)

  def test_training_failure_logged_not_raised_others_still_applied(self):
    p = FakeParams(fail_keys={"CompletedTrainingVersion"})
    apply_auto_onboarding(p, self.flag)  # must not raise
    self.assertEqual(p.store["HasAcceptedTerms"], terms_version)
    self.assertIs(p.store["SunnylinkEnabled"], True)
    self.assertIs(p.store["RecordFront"], False)

  def test_other_failures_do_not_block_training(self):
    p = FakeParams(fail_keys={"HasAcceptedTerms", "SunnylinkEnabled", "RecordFront"})
    apply_auto_onboarding(p, self.flag)
    self.assertEqual(p.store["CompletedTrainingVersion"], training_version)

  def test_training_version_is_real_constant(self):
    src = (REPO / "common" / "version.py").read_text()
    self.assertIn(f'training_version: str = "{training_version}"', src)
    self.assertNotEqual(training_version, "0")


class TestMiciOnboardingSkipsDM(unittest.TestCase):
  def test_onboarding_window_never_pushes_guide_unless_write_failed(self):
    ui = MICI_ONBOARD.read_text()
    win = ui[ui.index("class OnboardingWindow(Widget):"):]
    init = win[:win.index("def _mark_training_done")]
    self.assertIn("if not self._training_done:\n      self._mark_training_done()", init)
    mark = win[win.index("def _mark_training_done"):win.index("def _skip_or_show_training")]
    self.assertIn('ui_state.params.put("CompletedTrainingVersion", training_version, block=True)', mark)
    self.assertIn("except Exception", mark)
    self.assertIn("cloudlog.exception", mark)
    skip = win[win.index("def _skip_or_show_training"):win.index("def _on_uninstall")]
    self.assertIn("if self._training_done or self._mark_training_done():\n      self.close()", skip)
    # The only guide pushes are the write-failure fallbacks.
    self.assertEqual(win.count("gui_app.push_widget(self._training_guide)"), 2)
    for cb in ("def _on_terms_accepted", "def _on_sunnylink_accepted", "def _on_sunnylink_declined"):
      body = win[win.index(cb):]
      body = body[:body.index("\n  def ", 1)]
      self.assertIn("self._skip_or_show_training()", body)
      self.assertNotIn("push_widget(self._training_guide)", body)
    render = win[win.index("  def _render"):]
    self.assertIn("devc4-test2: only if auto-complete failed", render)

  def test_dm_processes_untouched_and_enforcement_disabled(self):
    policy = (REPO / "selfdrive" / "monitoring" / "policy.py").read_text()
    self.assertIn("DM_ENFORCEMENT_DISABLED = True", policy)


class TestRealKeysAndWiring(unittest.TestCase):
  def test_keys_exist_with_expected_types(self):
    keys = PARAMS_KEYS.read_text()
    for key, typ in (("HasAcceptedTerms", "STRING"), ("HasAcceptedTermsSP", "STRING"),
                     ("CompletedSunnylinkConsentVersion", "STRING"), ("SunnylinkEnabled", "BOOL"),
                     ("CompletedTrainingVersion", "STRING"), ("RecordFrontLock", "BOOL")):
      self.assertRegex(keys, rf'\{{"{key}", \{{PERSISTENT, {typ}')

  def test_gates_compare_against_these_versions(self):
    ui = MICI_ONBOARD.read_text()
    self.assertIn('ui_state.params.get("HasAcceptedTerms") == terms_version', ui)
    self.assertIn('ui_state.params.get("HasAcceptedTermsSP") == terms_version_sp', ui)
    self.assertIn('ui_state.params.get("CompletedSunnylinkConsentVersion") in {', ui)
    hw = HARDWARED.read_text()
    self.assertIn('params.get("HasAcceptedTerms") == terms_version', hw)
    self.assertIn('params.get("HasAcceptedTermsSP") == terms_version_sp', hw)
    self.assertIn('params.get("CompletedTrainingVersion") == training_version', hw)
    self.assertTrue(training_version)

  def test_manager_calls_it_after_defaults(self):
    mgr = MANAGER.read_text()
    self.assertIn("from openpilot.sunnypilot.system.devc4_auto_onboarding import apply_auto_onboarding", mgr)
    self.assertIn("apply_auto_onboarding(params)", mgr)
    self.assertLess(mgr.index("params.get_default_value(k)"), mgr.index("apply_auto_onboarding(params)"))
    self.assertLess(mgr.index("apply_auto_onboarding(params)"), mgr.index("register(show_spinner=True)"))

  def test_attention_notice_auto_skipped(self):
    ui = MICI_ONBOARD.read_text()
    cls = ui[ui.index("class TrainingGuide(NavWidget):"):ui.index("class QRCodeWidget")]
    self.assertIn('TrainingGuideAttentionNotice(continue_callback=lambda: gui_app.push_widget(self._steps[1]))', cls)
    self.assertIn("self._auto_skip_attention_notice = True", cls)
    render = cls[cls.index("def _render"):]
    pattern = (r"if self\._auto_skip_attention_notice and self\.enabled and not self\.is_dismissing:\n" +
               r"\s+self\._auto_skip_attention_notice = False\n\s+gui_app\.push_widget\(self\._steps\[1\]\)")
    self.assertRegex(render, re.compile(pattern))
    # "driver camera data" (RecordFront) page skipped: the DM tutorial finishes the guide
    self.assertIn("TrainingGuideDMTutorial(continue_callback=completed_callback),", cls)
    self.assertNotIn("gui_app.push_widget(self._steps[3])", cls)
    self.assertIn("TrainingGuidePreDMTutorial(continue_callback=lambda: gui_app.push_widget(self._steps[2]))", cls)
    # its decline still writes False if ever shown; no default -> unset reads as off
    self.assertIn('ui_state.params.put_bool("RecordFront", False)', ui)
    self.assertRegex(PARAMS_KEYS.read_text(), r'\{"RecordFront", \{PERSISTENT \| BACKUP, BOOL\}\}')
    self.assertIn('ui_state.params.put("CompletedTrainingVersion", training_version, block=True)', ui)


if __name__ == "__main__":
  unittest.main()
