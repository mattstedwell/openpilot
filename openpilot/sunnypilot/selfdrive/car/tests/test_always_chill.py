"""devc4-test2: longitudinal always runs in Chill (ACC). The model's
e2e desiredAcceleration and shouldStop never feed the planner.

On this sunnypilot/dev tree, official ExperimentalMode is a deprecated
cereal field with no Settings toggle or Params key. ALWAYS_CHILL drops
the e2e candidate from the planner (min of lead-MPC and cruise only)
and forces the deprecated field off.

Host-safe: no Params / msgq / display.
"""
import inspect
import math
from pathlib import Path

from openpilot.cereal import log
from openpilot.sunnypilot.selfdrive.car.always_chill import ALWAYS_CHILL, experimental_mode_active

REPO = Path(__file__).resolve().parents[4]
PLANNER = REPO / "selfdrive" / "controls" / "lib" / "longitudinal_planner.py"
SELFDRIVED = REPO / "selfdrive" / "selfdrived" / "selfdrived.py"
PARAMS_KEYS = REPO / "common" / "params_keys.h"
LONG_INDICATOR = REPO / "selfdrive" / "ui" / "mici" / "onroad" / "long_indicator.py"
MODEL_RENDERER = REPO / "selfdrive" / "ui" / "mici" / "onroad" / "model_renderer.py"
ALWAYS_CHILL_PY = REPO / "sunnypilot" / "selfdrive" / "car" / "always_chill.py"

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource


def _load_choose_acc_plan():
  src = PLANNER.read_text()
  start = src.index("def choose_acc_plan(")
  end = src.index("\nclass LongitudinalPlanner")
  glb = {"LongitudinalPlanSource": LongitudinalPlanSource}
  exec(src[start:end], glb)
  return glb["choose_acc_plan"]


choose_acc_plan = _load_choose_acc_plan()


def test_flag_and_gate():
  assert ALWAYS_CHILL is True
  assert experimental_mode_active(True) is False
  assert experimental_mode_active(False) is False
  assert experimental_mode_active() is False


def test_no_lead_model_stop_follows_cruise_only():
  """No lead, model planning a hard stop: output is cruise, not the model.

  a_mpc is set above cruise, as a fake far-and-faster lead that is not
  constraining. The model's desiredAcceleration / shouldStop would have
  won the official 3-way min() and forced a stop; they are not candidates.
  """
  a, source, should_stop = choose_acc_plan(
      1.5, LongitudinalPlanSource.lead0, False,
      0.7, False)
  assert math.isclose(a, 0.7)
  assert source == LongitudinalPlanSource.cruise
  assert should_stop is False
  assert list(inspect.signature(choose_acc_plan).parameters) == [
      "a_mpc", "source_mpc", "stop_mpc", "a_cruise", "stop_cruise"]


def test_no_lead_model_should_stop_does_not_stop():
  """Model shouldStop alone must not put the car into a stop."""
  a, source, should_stop = choose_acc_plan(
      0.4, LongitudinalPlanSource.lead0, False,
      0.8, False)
  assert should_stop is False
  assert a > 0.0


def test_with_lead_follows_the_lead():
  """A lead-MPC request more restrictive than cruise wins; a harsher
  (imagined) model request is not in the blend."""
  a, source, should_stop = choose_acc_plan(
      -1.2, LongitudinalPlanSource.lead0, False,
      0.8, False)
  assert math.isclose(a, -1.2)
  assert source == LongitudinalPlanSource.lead0
  assert should_stop is False

  a, source, should_stop = choose_acc_plan(
      -0.5, LongitudinalPlanSource.lead1, True,
      0.4, False)
  assert math.isclose(a, -0.5)
  assert source == LongitudinalPlanSource.lead1
  assert should_stop is True


def test_cruise_can_still_stop_without_a_model():
  """forceDecel / lowering set speed can still stop via the cruise path."""
  a, source, should_stop = choose_acc_plan(
      0.3, LongitudinalPlanSource.lead0, False,
      -0.4, True)
  assert math.isclose(a, -0.4)
  assert source == LongitudinalPlanSource.cruise
  assert should_stop is True


def test_equal_accel_prefers_lead_mpc_first():
  a, source, should_stop = choose_acc_plan(
      0.5, LongitudinalPlanSource.lead0, False,
      0.5, False)
  assert math.isclose(a, 0.5)
  assert source == LongitudinalPlanSource.lead0
  assert should_stop is False


def test_planner_uses_acc_only_when_always_chill():
  src = PLANNER.read_text()
  assert "from openpilot.sunnypilot.selfdrive.car.always_chill import ALWAYS_CHILL" in src
  assert "def choose_acc_plan(" in src
  assert "if ALWAYS_CHILL:" in src
  i_chill = src.index("if ALWAYS_CHILL:")
  i_else = src.index("else:", i_chill)
  chill = src[i_chill:i_else]
  # Official 3-way blend is the else of ALWAYS_CHILL; the personality
  # cap clip (this tree's stock ACCEL_MAX, Relaxed 80%, etc.) follows it.
  official = src[i_else:src.index("self.output_a_target = np.clip(output_a_target, ACCEL_MIN, cap)")]
  assert "choose_acc_plan(" in chill
  assert "output_a_target_e2e" not in chill
  assert "LongitudinalPlanSource.e2e" not in chill
  assert "sm['modelV2'].action" not in chill
  # Official Experimental blend is still in the file, behind ALWAYS_CHILL False
  assert "output_a_target_e2e" in official
  assert "LongitudinalPlanSource.e2e" in official
  assert "sm['modelV2'].action.desiredAcceleration" in official
  assert "sm['modelV2'].action.shouldStop" in official


def test_experimental_mode_always_off():
  sd = SELFDRIVED.read_text()
  assert "from openpilot.sunnypilot.selfdrive.car.always_chill import ALWAYS_CHILL" in sd
  assert "ss.deprecated.experimentalMode = False" in sd
  assert "if ALWAYS_CHILL:" in sd
  # never turned on from a param
  assert 'get("ExperimentalMode"' not in sd
  assert 'get_bool("ExperimentalMode"' not in sd
  keys = PARAMS_KEYS.read_text()
  assert "ExperimentalMode" not in keys


def test_experimental_hud_indicator_unreachable():
  ind = LONG_INDICATOR.read_text()
  assert "ALWAYS_CHILL" in ind
  assert "e2e = (not ALWAYS_CHILL) and has_lead" in ind
  rnd = MODEL_RENDERER.read_text()
  assert "ALWAYS_CHILL" in rnd
  assert "(not ALWAYS_CHILL) and plan.longitudinalPlanSource == log.LongitudinalPlan.LongitudinalPlanSource.e2e" in rnd
  assert "radar.leadOne" in rnd and "radar.leadTwo" in rnd


def test_always_chill_module():
  src = ALWAYS_CHILL_PY.read_text()
  assert "ALWAYS_CHILL = True" in src
  assert "def experimental_mode_active" in src
