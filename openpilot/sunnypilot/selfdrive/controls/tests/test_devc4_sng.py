"""dev-c4-sng-test: automatic stop-and-go / city smoothing.

Unit tests for devc4_sng (no solver) plus closed-loop host sims
(sng_city_sim.py: real MPC + LongControl + Toyota TSS2 PCM emulation)
comparing this tree's default StopAndGo against dev-c4 9c6e321c06
behaviour (module off, no brake-hold release). Closed-loop parts need
the host acados build and are skipped without it. Not road-tested.
"""
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib import devc4_sng as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PKG = ROOT / "openpilot"
MPH = 0.44704
DT = 0.05


def lead(d=10.0, v=2.0, a=0.0, present=True):
  return SimpleNamespace(status=present, present=present, dRel=d, vLead=v, aLeadK=a)


def run_detector(det, steps, v, ld):
  w = 0.0
  for _ in range(steps):
    w = det.update(v, ld, DT)
  return w


# --- detector ---------------------------------------------------------------

def test_not_active_on_open_road_or_highway_follow():
  det = S.StopAndGoDetector()
  assert run_detector(det, 200, 40 * MPH, lead(d=40, v=40 * MPH)) == 0.0
  assert run_detector(det, 200, 15 * MPH, lead(present=False)) == 0.0
  # 15 mph behind a 15 mph lead, no stop yet: not crawling, not active
  assert run_detector(det, 200, 15 * MPH, lead(d=15, v=15 * MPH)) == 0.0


def test_enters_after_stop_behind_lead_and_blends_in():
  det = S.StopAndGoDetector()
  w0 = run_detector(det, 2, 0.0, lead(d=8, v=0.0))
  assert det.active and w0 <= 2 * S.SNG_BLEND_RATE * DT + 1e-9
  ws = [det.update(15 * MPH, lead(d=15, v=15 * MPH), DT) for _ in range(60)]
  assert all(b - a <= S.SNG_BLEND_RATE * DT + 1e-9 for a, b in zip([w0] + ws, ws))
  assert ws[-1] == pytest.approx(1.0)


def test_enters_when_crawling():
  det = S.StopAndGoDetector()
  assert run_detector(det, 60, 6 * MPH, lead(d=10, v=7 * MPH)) == pytest.approx(1.0)


def test_exits_above_30mph_held_or_no_lead():
  det = S.StopAndGoDetector()
  run_detector(det, 60, 0.0, lead(d=8, v=0.0))
  run_detector(det, int(S.SNG_EXIT_HOLD_S / DT) - 2, 32 * MPH, lead(d=30, v=32 * MPH))
  assert det.active  # not held long enough yet
  run_detector(det, 5, 32 * MPH, lead(d=30, v=32 * MPH))
  assert not det.active
  det = S.StopAndGoDetector()
  run_detector(det, 60, 0.0, lead(d=8, v=0.0))
  run_detector(det, int(S.SNG_NO_LEAD_EXIT_S / DT) + 2, 10 * MPH, lead(present=False))
  assert not det.active


def test_detector_fails_closed_on_nan():
  det = S.StopAndGoDetector()
  run_detector(det, 60, 0.0, lead(d=8, v=0.0))
  assert det.update(float("nan"), lead(), DT) == 0.0 and not det.active


# --- outputs only ever make the plan gentler / more careful ------------------

@pytest.mark.parametrize("base_jf", [0.5, 1.0, 1.3])
def test_outputs_bounded(base_jf):
  sng = S.StopAndGo()
  cases = [(0.0, lead(d=7.5, v=0.0)), (3.0, lead(d=8, v=3.0, a=0.5)), (3.0, lead(d=6, v=1.0, a=-3.0)),
           (8.0, lead(d=15, v=8.0)), (15.0, lead(d=25, v=15.0)), (25.0, lead(d=40, v=25.0)), (5.0, lead(present=False))]
  for _ in range(3):
    for v, ld in cases:
      for plan_min in (0.0, -1.0, -3.0, None):
        o = sng.update(v, ld, base_jf, 1.5, 1.05, 9.0, 0.3, False, DT, base_t_follow=1.25, plan_a_min=plan_min)
        assert o.cap <= 1.5 + 1e-9
        assert o.jerk_factor >= base_jf - 1e-9           # never less smoothing than the personality
        assert o.danger_factor is None or o.danger_factor >= S.LEAD_DANGER_FACTOR_STOCK
        assert o.danger_cost is None or o.danger_cost >= S.DANGER_ZONE_COST_STOCK
        assert 0.0 <= o.stop_extra_predicted <= 1.0
        assert o.t_follow is None                         # default: personality follow time unchanged
        assert not o.brake_free


def test_relaxed_braking_floor_and_urgency_gate():
  sng = S.StopAndGo()
  for _ in range(60):
    o = sng.update(8.0, lead(d=15, v=8.0), 0.5, 1.5, 1.05, 12.0, 0.0, False, DT, 1.25, plan_a_min=-0.3)
  assert o.jerk_factor == pytest.approx(S.RELAXED_JERK_FLOOR)   # Aggressive brakes like Relaxed
  o = sng.update(8.0, lead(d=15, v=8.0, a=-3.0), 0.5, 1.5, 1.05, 12.0, 0.0, False, DT, 1.25, plan_a_min=-0.3)
  assert o.jerk_factor == pytest.approx(0.5)                    # hard-braking lead: personality's own response
  o = sng.update(8.0, lead(d=15, v=8.0), 0.5, 1.5, 1.05, 12.0, 0.0, False, DT, 1.25, plan_a_min=-3.0)
  assert o.jerk_factor == pytest.approx(0.5)                    # urgent plan: same
  o = sng.update(8.0, lead(d=15, v=8.0), 0.5, 1.5, 1.05, 12.0, 0.0, True, DT, 1.25, plan_a_min=-0.3)
  assert o.jerk_factor == pytest.approx(0.5)                    # FCW: same


def test_city_weight_fades_out_by_55mph_and_needs_lead():
  assert S.city_weight(40 * MPH, lead()) == 1.0
  assert S.city_weight(50 * MPH, lead()) == pytest.approx(0.5)
  assert S.city_weight(56 * MPH, lead()) == 0.0
  assert S.city_weight(30 * MPH, lead(present=False)) == 0.0
  sng = S.StopAndGo()
  o = sng.update(70 * MPH, lead(d=60, v=70 * MPH), 0.5, 0.5, 0.45, 60.0, 0.0, False, DT, 1.25, plan_a_min=0.0)
  assert o.jerk_factor == 0.5 and o.cap == 0.5 and o.danger_factor == S.LEAD_DANGER_FACTOR_STOCK
  assert o.stop_extra_predicted == 0.0


def test_disabled_config_is_passthrough():
  sng = S.StopAndGo(S.SngConfig(enabled=False))
  o = sng.update(0.0, lead(d=7.5, v=0.0), 0.5, 1.5, 1.05, 7.5, 0.0, False, DT)
  assert o.jerk_factor is None and o.cap == 1.5 and o.danger_factor is None and o.danger_cost is None


def test_default_config_is_the_chosen_setup():
  c = S.DEFAULT_CONFIG
  assert c.enabled and c.city and c.relaxed_cap and c.stop_extra_predicted and c.urgency_gate
  assert c.jerk_floor == 1.0 and c.danger_factor == 1.0 and c.danger_cost_scale >= 1.0
  assert not c.safe_launch and not c.brake_onset_free
  assert c.t_follow_min is None and c.traffic_t_follow is None and c.traffic_jerk_scale is None


# --- source tripwires --------------------------------------------------------

def test_planner_and_mpc_wiring():
  src = (PKG / "selfdrive/controls/lib/longitudinal_planner.py").read_text()
  assert "self.sng = StopAndGo()" in src
  assert "cap = min(cap, sng.cap)" in src
  assert "jerk_factor=jerk_factor, danger_cost=danger_cost" in src
  assert "self.output_a_target = np.clip(output_a_target, ACCEL_MIN, cap)" in src
  mpc = (PKG / "selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py").read_text()
  assert "max(LEAD_DANGER_FACTOR, float(self.lead_danger_factor))" in mpc
  assert "max(DANGER_ZONE_COST, float(danger_cost))" in mpc
  assert "np.maximum(pred - cur, 0.0)" in mpc
  assert "STOP_DISTANCE = 6.0" in mpc


# --- closed loop --------------------------------------------------------------

def _sim(cand, pers):
  build = Path(os.environ.get("TEST3_MPC_BUILD", "/tmp/test3dev_mpc_build"))
  build.mkdir(parents=True, exist_ok=True)
  out = Path(tempfile.mkdtemp()) / "sng.json"
  env = dict(os.environ)
  extra = [str(ROOT), str(ROOT / "opendbc_repo"), str(ROOT / "msgq_repo")]
  env["PYTHONPATH"] = os.pathsep.join([p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p] + extra)
  subprocess.run([sys.executable, str(HERE / "sng_city_sim.py"), str(ROOT), str(build), str(out), json.dumps(cand), pers],
                 check=True, env=env, cwd=tempfile.gettempdir(), capture_output=True, timeout=1500)
  return json.loads(out.read_text())["cases"][pers]


@pytest.fixture(scope="module")
def sims():
  if importlib.util.find_spec("acados") is None:
    pytest.skip("host acados build not available")
  jobs = {(c, p): (cand, p) for c, cand in (("on", {"name": "default", "sng": {}, "release": True}),
                                            ("off", {"name": "dev-c4", "sng": None, "release": False}))
          for p in ("aggressive", "standard", "relaxed")}
  with ThreadPoolExecutor(max_workers=6) as ex:
    futs = {k: ex.submit(_sim, *v) for k, v in jobs.items()}
    return {k: f.result() for k, f in futs.items()}


PERS = ["aggressive", "standard", "relaxed"]


@pytest.mark.parametrize("p", PERS)
def test_no_collisions_and_never_closer_than_dev_c4(sims, p):
  on, off = sims[("on", p)], sims[("off", p)]
  for name, r in on.items():
    assert not r["collision"], f"{p} {name}: collision"
    assert r["min_gap_m"] >= off[name]["min_gap_m"] - 0.05, (p, name, r["min_gap_m"], off[name]["min_gap_m"])
    if r["standstill_gap_minus_stop_gap_m"] is not None and off[name]["standstill_gap_minus_stop_gap_m"] is not None:
      assert r["standstill_gap_minus_stop_gap_m"] >= off[name]["standstill_gap_minus_stop_gap_m"] - 0.05, (p, name)
    assert r["solve_fail"] == 0, (p, name)


@pytest.mark.parametrize("p", PERS)
def test_stops_at_set_gap_when_lead_stops_normally(sims, p):
  on = sims[("on", p)]
  # Ordinary stop-and-go / creeps / lights: standstill gap within 0.3 m (1 ft) of the set gap.
  for name in ("accordion_0_15mph", "creeps_1.5_3_1m", "false_start", "city_lights_40mph", "stopped_lead_hold"):
    assert on[name]["standstill_gap_minus_stop_gap_m"] >= -0.3, (p, name, on[name]["standstill_gap_minus_stop_gap_m"])


@pytest.mark.parametrize("p", PERS)
def test_gentler_not_hesitant(sims, p):
  on, off = sims[("on", p)], sims[("off", p)]
  for name, r in on.items():
    assert r["peak_accel"] <= off[name]["peak_accel"] + 0.05, (p, name)  # never more accel
    if off[name]["t_move_mean"] is not None:
      assert r["t_move_max"] <= off[name]["t_move_max"] + 0.05, (p, name)  # never slower to move
  for name in ("accordion_0_15mph", "accordion_noisy_radar", "city_lights_40mph"):
    assert on[name]["peak_decel"] >= off[name]["peak_decel"] - 0.05, (p, name)  # Relaxed-style, not harder
    # the Relaxed floor + set-gap keeping trade a little RMS jerk in noisy stops for the gap (sims: <= +0.05)
    assert on[name]["rms_jerk"] <= off[name]["rms_jerk"] + 0.08, (p, name)


@pytest.mark.parametrize("p", PERS)
def test_stopped_lead_never_moves(sims, p):
  r = sims[("on", p)]["stopped_lead_hold"]
  assert r["v_ego_peak_mph"] == 0.0 and r["t_move"] == []
