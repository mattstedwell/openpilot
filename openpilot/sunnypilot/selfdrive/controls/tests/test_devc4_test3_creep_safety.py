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


def _run(cases, ablate=""):
  if importlib.util.find_spec("acados") is None:
    pytest.skip("host acados build not available")
  build = Path(os.environ.get("TEST3_MPC_BUILD", "/tmp/test3dev_mpc_build"))
  build.mkdir(parents=True, exist_ok=True)
  out = Path(tempfile.mkdtemp()) / "creep.json"
  env = dict(os.environ)
  env["SAFE_LAUNCH_ABLATE"] = ablate
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


SAFE_LAUNCH_CASES = "creep,to3mph,to6mph,to10mph,launch_up,stopped"


@pytest.fixture(scope="module")
def safe_launch_on():
  return _run(SAFE_LAUNCH_CASES)


@pytest.fixture(scope="module")
def safe_launch_off():
  # Same tree with the t309u safe launch (MPC weighting + brake-hold
  # release) switched off in the sim: that is e9e85baf79 / abab191b19.
  return _run(SAFE_LAUNCH_CASES, ablate="nompc,nocc")


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


# --- t309u safe launch ------------------------------------------------------

@pytest.mark.parametrize("personality", ["aggressive", "standard", "relaxed"])
def test_safe_launch_never_closer_than_before(safe_launch_on, safe_launch_off, personality):
  on, off = safe_launch_on[personality], safe_launch_off[personality]
  for name, r in on.items():
    if name.startswith("launch_up"):
      continue
    assert not r["collision"], f"{personality} {name}: collision"
    assert r["min_gap_m"] >= off[name]["min_gap_m"] - 0.02, \
      f"{personality} {name}: min gap {r['min_gap_m']} m vs {off[name]['min_gap_m']} m without safe launch"
    if not name.startswith("stopped"):
      assert r["v_ego_peak_mph"] <= r["lead_peak_mph"], f"{personality} {name}: ego out-ran the lead"


@pytest.mark.parametrize("personality", ["aggressive", "standard", "relaxed"])
def test_safe_launch_is_quicker(safe_launch_on, safe_launch_off, personality):
  on, off = safe_launch_on[personality], safe_launch_off[personality]
  for name in ("launch_up1.0_to25mph_hold", "launch_up2.0_to25mph_hold"):
    assert on[name]["t_move_005"] <= off[name]["t_move_005"] - 0.2, (name, on[name]["t_move_005"], off[name]["t_move_005"])
    # dev-c4 comfort build limits accel build-up (0.45 m/s^3), so reaching
    # 1 mph gains a little less than on test3. dev-c4-sng-test: SafeLaunch
    # MPC weighting is off by default (brake-hold release only), which gains
    # ~0.09-0.2 s to 1 mph; still must be quicker.
    assert on[name]["t_move_1mph"] <= off[name]["t_move_1mph"] - 0.05, (name, on[name]["t_move_1mph"], off[name]["t_move_1mph"])
    # Never pushes toward the cap: peak accel no higher than before (+ noise).
    assert on[name]["peak_accel"] <= off[name]["peak_accel"] + 0.1


@pytest.mark.parametrize("personality", ["aggressive", "standard", "relaxed"])
def test_safe_launch_stopped_lead_never_moves(safe_launch_on, personality):
  r = safe_launch_on[personality]["stopped_lead_hold"]
  assert r["t_move_005"] is None and r["v_ego_peak_mph"] == 0.0
  assert r["min_gap_m"] == pytest.approx(r["gap0_ft"] * 0.3048, abs=0.02)


# --- dev-c4-sng-test: StopAndGo + release vs dev-c4 9c6e321c06 ----------------

@pytest.fixture(scope="module")
def dev_c4_baseline():
  # No StopAndGo, no SafeLaunch, no brake-hold release: dev-c4 9c6e321c06.
  return _run(SAFE_LAUNCH_CASES + "," + MATT_CASES, ablate="nosng,nompc,nocc")


@pytest.fixture(scope="module")
def sng_on():
  return _run(SAFE_LAUNCH_CASES + "," + MATT_CASES)


@pytest.mark.parametrize("personality", ["aggressive", "standard", "relaxed"])
def test_sng_creep_never_closer_or_faster_than_dev_c4(sng_on, dev_c4_baseline, personality):
  on, off = sng_on[personality], dev_c4_baseline[personality]
  for name, r in on.items():
    assert not r["collision"], f"{personality} {name}: collision"
    assert r["min_gap_m"] >= off[name]["min_gap_m"] - 0.05, (personality, name, r["min_gap_m"], off[name]["min_gap_m"])
    assert r["peak_accel"] <= off[name]["peak_accel"] + 0.05, (personality, name)
    if not name.startswith(("stopped", "launch_up")):
      assert r["v_ego_peak_mph"] <= r["lead_peak_mph"], f"{personality} {name}: ego out-ran the lead"
