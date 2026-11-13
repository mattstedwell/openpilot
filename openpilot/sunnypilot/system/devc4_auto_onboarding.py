"""devc4-test2: auto-accept terms and auto-enable sunnylink from manager_init.

Terms (every boot, so future terms version bumps are covered too):
  HasAcceptedTerms   -> system.version.terms_version     (read by UI onboarding
  HasAcceptedTermsSP -> system.version.terms_version_sp   and hardwared startup)
Only written when the stored value differs.

sunnylink:
  - Once per device (one-time flag file beside the params store, same scheme
    as listed defaults): SunnylinkEnabled = True and
    CompletedSunnylinkConsentVersion = sunnylink_consent_version.
    After that SunnylinkEnabled is never touched again, so turning sunnylink
    off in Settings sticks.
  - Every boot: if CompletedSunnylinkConsentVersion is neither the current
    consent version nor "declined" (e.g. a future consent version bump), it
    is set to the current version so the consent page never blocks
    onboarding. SunnylinkEnabled is not changed by this step.

Driver camera data sharing (RecordFront, the mici training guide's "driver
camera data" page, which that build no longer shows): declined by storing
RecordFront = False only when it has never been set (None). Unset already
reads as off (no default in params_keys.h), so this only makes the decline
explicit; a stored choice, including a later opt-in in Settings, is never
changed, and RecordFrontLock (internal fleet, forces it on in manager_init)
is respected.

Training guide (CompletedTrainingVersion -> system.version.training_version,
every boot, only written when the stored value differs, so future training
version bumps are covered too): this skips the comma 4 "driver monitoring
check" intro and the face (DM camera) check during first setup. hardwared's
onroad "completed_training" startup check reads the same value. Driver
monitoring itself (dmonitoringd/dmonitoringmodeld) is not changed.

Every param write is wrapped: errors are logged and skipped, never raised,
so manager_init cannot crash because of this helper.
"""
from __future__ import annotations

import os

from openpilot.common.version import (
  terms_version, terms_version_sp, training_version, sunnylink_consent_version, sunnylink_consent_declined,
)
from openpilot.sunnypilot.system.devc4_listed_defaults import (
  _flag_already_present, _mark_migrated, _log_flag_oserror,
)

SUNNYLINK_FLAG_NAME = "devc4_test2_sunnylink_enable_v1"


def _log(msg: str) -> None:
  try:
    from openpilot.common.swaglog import cloudlog
    cloudlog.exception(f"devc4-test2 auto onboarding: {msg}")
  except Exception:
    import sys
    print(f"WARNING: devc4-test2 auto onboarding: {msg}", file=sys.stderr)


def _get(params, key):
  try:
    return params.get(key)
  except Exception as err:
    _log(f"get {key} failed: {err}")
    return None


def _put(params, key, value) -> bool:
  try:
    if isinstance(value, bool):
      params.put_bool(key, value, block=True)
    else:
      params.put(key, value, block=True)
    return True
  except Exception as err:
    _log(f"put {key} failed: {err}")
    return False


def _ensure(params, key, value) -> bool:
  """Write value if the stored one differs. Returns True if it is in place."""
  if _get(params, key) == value:
    return True
  return _put(params, key, value)


def default_sunnylink_flag_path() -> str:
  from openpilot.common.hardware.hw import Paths
  return os.path.join(Paths.params(), SUNNYLINK_FLAG_NAME)


def auto_accept_terms(params) -> None:
  _ensure(params, "HasAcceptedTerms", terms_version)
  _ensure(params, "HasAcceptedTermsSP", terms_version_sp)


def auto_complete_training(params) -> bool:
  """Returns True if CompletedTrainingVersion is the current training_version."""
  return _ensure(params, "CompletedTrainingVersion", training_version)


def auto_enable_sunnylink(params, flag_path: str | None = None) -> bool:
  """Returns True if the one-time enable ran (and stuck) this call."""
  ran = False
  if flag_path is None:
    try:
      flag_path = default_sunnylink_flag_path()
    except Exception as err:
      _log_flag_oserror("resolve path", "<sunnylink flag path>", err)
      flag_path = None

  present = _flag_already_present(flag_path) if flag_path else None
  if present is False:
    ok = _put(params, "SunnylinkEnabled", True)
    ok = _put(params, "CompletedSunnylinkConsentVersion", sunnylink_consent_version) and ok
    # flag only once both writes landed, so a failed boot retries next time
    if ok and _mark_migrated(flag_path):
      ran = True

  if _get(params, "CompletedSunnylinkConsentVersion") not in (sunnylink_consent_version, sunnylink_consent_declined):
    _put(params, "CompletedSunnylinkConsentVersion", sunnylink_consent_version)
  return ran


def auto_decline_record_front(params) -> bool:
  """Returns True if RecordFront was set to False (declined) this call."""
  if _get(params, "RecordFrontLock"):
    return False
  if _get(params, "RecordFront") is not None:
    return False
  return _put(params, "RecordFront", False)


def apply_auto_onboarding(params, sunnylink_flag_path: str | None = None) -> None:
  """Never raises."""
  try:
    auto_accept_terms(params)
  except Exception as err:
    _log(f"terms failed: {err}")
  try:
    auto_enable_sunnylink(params, sunnylink_flag_path)
  except Exception as err:
    _log(f"sunnylink failed: {err}")
  try:
    auto_decline_record_front(params)
  except Exception as err:
    _log(f"record front failed: {err}")
  try:
    auto_complete_training(params)
  except Exception as err:
    _log(f"training guide failed: {err}")
