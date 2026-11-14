"""Regression: stop-and-go creep near-collision (t305u / t307u / t308u).

On devc4-test3 d79496cac9 Matt reported (and had to brake for) this:
both cars stopped, the lead crept ~1.5-3 m (5-10 ft) and stopped again,
ego started moving and did not slow down. Host closed-loop sims showed
the LeadLaunch take-up (from 82083b36fc) ramping ego past the lead's
own speed toward the personality cap and holding a positive command
(soft-exit floor -0.5 m/s², 0.4 m/s³ jerk down) after the lead braked;
5ea5397a5c / d79496cac9 then weakened and rate-limited the brake
build-up below ~5 mph (FinalStopTaper envelope, longcontrol stopping
easing, 0.30 m/s³ ACCEL_CMD limit). Those changes are reverted here back
to abab191b19 behavior.

The closed-loop part runs this tree's real MPC, planner pieces, and
LongControl with a Toyota TSS2 carcontroller emulation and needs the
host acados build; it is skipped when acados is not importable.
Not road-tested.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]  # worktree root (has openpilot/, opendbc_repo/, msgq_repo/)
PKG = ROOT / "openpilot"
OPENDBC = ROOT / "opendbc_repo" / "opendbc"


# --- fast tripwires (no solver) -------------------------------------------

def test_planner_has_no_post_mpc_take_up_or_taper():
  src = (PKG / "selfdrive/controls/lib/longitudinal_planner.py").read_text()
  for name in ("LeadLaunch", "FinalStopTaper", "LeadSlow", "lead_launch", "final_stop", "lead_slow"):
    assert name not in src, f"{name} is back in the planner; re-run the creep regression first"
  assert "self.output_a_target = np.clip(output_a_target, ACCEL_MIN, cap)" in src


def test_toyota_accel_cmd_has_no_comfort_rate_limit():
  src = (OPENDBC / "car/toyota/carcontroller.py").read_text()
  assert "smooth_pcm_accel" not in src
  assert "fade_stopping_integral" not in src
  assert not (OPENDBC / "sunnypilot/car/toyota/smooth_stop_hold.py").exists()


def test_longcontrol_stopping_uses_official_ramp():
  src = (PKG / "selfdrive/controls/lib/longcontrol.py").read_text()
  assert "stopping_accel_step" not in src
  assert "if output_accel > self.CP.stopAccel:" in src


# --- closed loop -----------------------------------------------------------

MATT_CASES = "creep1.5m,creep2.0m,creep3.0m"
GRID_CASES = "to3mph,to6mph,to10mph"


def _run(cases):
  if importlib.util.find_spec("acados") is None:
    pytest.skip("host acados build not available")
  build = Path(os.environ.get("TEST3_MPC_BUILD", "/tmp/test3dev_mpc_build"))
  build.mkdir(parents=True, exist_ok=True)
  out = Path(tempfile.mkdtemp()) / "creep.json"
  env = dict(os.environ)
  extra = [str(ROOT), str(ROOT / "opendbc_repo"), str(ROOT / "msgq_repo")]
  env["PYTHONPATH"] = os.pathsep.join([p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p] + extra)
  subprocess.run([sys.executable, str(HERE / "test3_creep_safety_sim.py"), str(ROOT), str(build), str(out),
                  "", "aggressive,standard,relaxed", cases], check=True, env=env, cwd=tempfile.gettempdir(),
                 capture_output=True, timeout=1200)
  return json.loads(out.read_text())["cases"]


@pytest.fixture(scope="module")
def matt_results():
  return _run(MATT_CASES)


@pytest.fixture(scope="module")
def grid_results():
  return _run(GRID_CASES)


@pytest.mark.parametrize("personality", ["aggressive", "standard", "relaxed"])
def test_short_creep_then_stop_keeps_safe_gap(matt_results, personality):
  # t308u: lead creeps 1.5-3 m to ~3-5 mph, then stops.
  for name, r in matt_results[personality].items():
    assert not r["collision"], f"{personality} {name}: collision"
    assert r["min_gap_m"] >= 4.5, f"{personality} {name}: min gap {r['min_gap_m']} m"
    # Ego must never out-run a creeping lead.
    assert r["v_ego_peak_mph"] <= r["lead_peak_mph"], f"{personality} {name}: ego {r['v_ego_peak_mph']} mph"


@pytest.mark.parametrize("personality", ["aggressive", "standard", "relaxed"])
def test_creep_to_city_speed_then_brake_no_collision(grid_results, personality):
  # Lead 0.5-1.5 m/s² up to 3-10 mph, then -1.5 / -3.0 m/s² to a stop.
  for name, r in grid_results[personality].items():
    assert not r["collision"], f"{personality} {name}: collision"
    assert r["min_gap_m"] >= 3.5, f"{personality} {name}: min gap {r['min_gap_m']} m"
    assert r["v_ego_peak_mph"] <= r["lead_peak_mph"] + 0.5, f"{personality} {name}: ego {r['v_ego_peak_mph']} mph"
