"""Fork defaults applied from manager_init for existing Params keys only.

Prebuilt libparams_c.so keeps official compiled defaults, so these values are
applied here. Missing ALC and on-road screen params are filled every boot.
Official stored ALC and screen defaults are migrated once, then left alone
so a later user change is not overwritten.

  - AutoLaneChangeTimer 2 (0.5 s) and AutoLaneChangeBsmDelay on: missing keys
    are filled; official Timer=0 / BSM off are migrated once.
  - OnroadScreenOffBrightness 2 (Screen Off) and OnroadScreenOffTimer 3:
    official compiled defaults are Auto (0) + 15 s. Also migrate the previous
    unlisted 5 s Screen Off default once. Official alert-wake is unchanged.
    Screensaver is not changed.

SshEnabled, AdbEnabled, and ChevronInfo are not filled or migrated. Official
compiled behavior is used (SSH/ADB unset/off, Chevron All). Stored device
values, including ones written by a previous build, are left alone.

The one-time flag lives under Python Paths.params(), which matches C++
Path::params() (PARAMS_ROOT, or ~/.comma$OPENPILOT_PREFIX/params on PC, or
/data/params on device). It is a sibling of the key directory (d/), not a
Params key and not inside that destructive key store. The flag name is
test-branch specific so an already-run devc4 flag cannot skip ALC or screen
migration. Filesystem errors around the flag are logged and skipped so they
cannot crash manager_init; the flag is never marked complete unless the
write succeeds.
"""
from __future__ import annotations

import os
import sys
import tempfile

# AutoLaneChangeTimer: 2 == AutoLaneChangeMode.HALF_SECOND == 0.5 s
# OnroadScreenOffBrightness: 2 == Screen Off (official 0 == Auto)
# OnroadScreenOffTimer: 3 seconds (official 15; a valid mapped timer value)
DEFAULT_OVERRIDES = {
  "AutoLaneChangeTimer": 2,
  "AutoLaneChangeBsmDelay": True,
  "OnroadScreenOffBrightness": 2,
  "OnroadScreenOffTimer": 3,
}

OFFICIAL_SCREEN_BRIGHTNESS = 0  # Auto
OFFICIAL_SCREEN_TIMER = 15
# Previous unlisted C4-only default (Screen Off + 5 s), migrated to listed 3 s.
PREV_SCREEN_BRIGHTNESS = 2
PREV_SCREEN_TIMER = 5

# Not a Params key: prebuilt libparams rejects unknown keys.
# Placed in Paths.params() (e.g. /data/params), beside d/ and .lock, not inside d/.
# New name for this branch so a prior devc4-test listed_defaults flag does not
# skip ALC or screen migration on a device that still has official defaults.
MIGRATION_FLAG_NAME = "devc4_test2_listed_defaults_v1"


# Patchable by host tests. Lazy-imported so importing this module does not
# pull openpilot.system.hardware (which needs device-only deps).
Paths = None


def _paths():
  global Paths
  if Paths is None:
    from openpilot.common.hardware.hw import Paths as _Paths
    Paths = _Paths
  return Paths


def default_migration_flag_path() -> str:
  return os.path.join(_paths().params(), MIGRATION_FLAG_NAME)


def _is_official_alc_timer(value) -> bool:
  return value in (None, 0)


def _is_official_bsm_delay(value) -> bool:
  return value in (None, 0, False)


def _is_migratable_screen(brightness, timer) -> bool:
  official = brightness in (None, OFFICIAL_SCREEN_BRIGHTNESS) and timer in (None, OFFICIAL_SCREEN_TIMER)
  previous_unlisted = brightness == PREV_SCREEN_BRIGHTNESS and timer == PREV_SCREEN_TIMER
  return official or previous_unlisted


def _put(params, key, value) -> None:
  if hasattr(params, "put"):
    try:
      params.put(key, value, block=True)
      return
    except TypeError:
      params.put(key, value)
      return
  raise AttributeError("params object has no put()")


def _log_flag_oserror(op: str, path: str, err: BaseException) -> None:
  msg = f"devc4-test2 listed defaults: {op} failed for {path}: {err}"
  try:
    from openpilot.common.swaglog import cloudlog
    cloudlog.exception(msg)
  except Exception:
    print(f"WARNING: {msg}", file=sys.stderr)


def _flag_already_present(flag_path: str) -> bool | None:
  """True/False if the flag can be stat'd; None if the filesystem check failed."""
  try:
    return os.path.exists(flag_path)
  except OSError as err:
    _log_flag_oserror("stat", flag_path, err)
    return None


def _mark_migrated(flag_path: str) -> bool:
  """Atomically write the flag. Returns True only if the flag is in place."""
  directory = os.path.dirname(flag_path) or "."
  fd = -1
  tmp = None
  try:
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_devc4_test2_defaults_", dir=directory)
    os.write(fd, b"1\n")
    os.close(fd)
    fd = -1
    os.replace(tmp, flag_path)
    tmp = None
    return True
  except OSError as err:
    _log_flag_oserror("write", flag_path, err)
    if fd >= 0:
      try:
        os.close(fd)
      except OSError:
        pass
    if tmp:
      try:
        os.unlink(tmp)
      except OSError:
        pass
    return False


def apply_listed_defaults(params, flag_path: str | None = None) -> bool:
  """Fill missing ALC/screen keys; one-time-migrate official defaults if needed.

  Returns True if a one-time official-default migration ran this call.
  Flag filesystem failures are swallowed after logging; other errors propagate.
  Does not read or write SshEnabled, AdbEnabled, or ChevronInfo.
  """
  # Apply missing-key defaults first, before any flag I/O, so a path problem
  # cannot skip filling missing keys and so the flag is not written first.
  for key, value in DEFAULT_OVERRIDES.items():
    if params.get(key) is None:
      _put(params, key, value)

  if flag_path is None:
    try:
      flag_path = default_migration_flag_path()
    except OSError as err:
      _log_flag_oserror("resolve path", "<flag path>", err)
      return False

  already_migrated = _flag_already_present(flag_path)
  if already_migrated is None:
    # Do not guess; do not rewrite stored values or claim completion.
    return False
  if already_migrated:
    return False

  migrated = False
  # Only rewrite official ALC defaults (Nudge + BSM off). A non-default
  # timer means the user already touched lane-change settings, so leave
  # BSM delay alone even if it is still off.
  timer = params.get("AutoLaneChangeTimer")
  official_timer = _is_official_alc_timer(timer)
  if official_timer:
    _put(params, "AutoLaneChangeTimer", DEFAULT_OVERRIDES["AutoLaneChangeTimer"])
    migrated = True
    if _is_official_bsm_delay(params.get("AutoLaneChangeBsmDelay")):
      _put(params, "AutoLaneChangeBsmDelay", DEFAULT_OVERRIDES["AutoLaneChangeBsmDelay"])
      migrated = True

  if _is_migratable_screen(params.get("OnroadScreenOffBrightness"), params.get("OnroadScreenOffTimer")):
    _put(params, "OnroadScreenOffBrightness", DEFAULT_OVERRIDES["OnroadScreenOffBrightness"])
    _put(params, "OnroadScreenOffTimer", DEFAULT_OVERRIDES["OnroadScreenOffTimer"])
    migrated = True

  # Only mark complete if the flag actually landed.
  _mark_migrated(flag_path)
  return migrated
