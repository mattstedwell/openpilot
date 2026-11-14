"""dev-c4-human: human-like city following.

Unit tests for devc4_human_city (no solver) plus closed-loop host sims
(human_city_sim.py: real acados MPC + LongControl + Toyota TSS2 PCM
emulation) comparing this tree against dev-c4 9c6e321c06 behaviour
(module off). Includes the Nov 13 near-miss family (lead creeps then
stops, starts then stops, creeps then brakes hard, appears late, drops
out briefly, cut-in at a crawl). Closed-loop parts need the host acados
build and are skipped without it. Not road-tested.
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

from openpilot.sunnypilot.selfdrive.controls.lib import devc4_human_city as HC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PKG = ROOT / "openpilot"
MPH = 0.44704
DT = 0.05
PERS = ["aggressive", "standard", "relaxed"]
STOCK_T = {"aggressive": 1.25, "standard": 1.45, "relaxed": 1.75}


def lead(d=15.0, v=10.0, a=0.0, present=True, tau=1.5):
  return SimpleNamespace(status=present, present=present, dRel=d, vLead=v, aLeadK=a, aLeadTau=tau, modelProb=1.0)


def run_hc(hc, steps, v, ld, pers="standard", base_t=1.45, fcw=False, plan_a_min=0.0):
  o = None
  for _ in range(steps):
    o = hc.update(v, ld, pers, 1.0, base_t, 1.2, 10.0, fcw, DT, plan_a_min=plan_a_min)
  return o


# --- dense follow ---------------------------------------------------------------

@pytest.mark.parametrize("mph", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
def test_dense_target_ordering_floor(mph):
  v = mph * MPH
  ts = [HC.dense_t_target(v, p) for p in PERS]
  assert ts[0] < ts[1] < ts[2]
  assert min(ts) >= HC.DENSE_T_FLOOR


def test_dense_never_above_stock_and_slow_to_shrink():
  for p in PERS:
    hc = HC.HumanCity()
    prev = STOCK_T[p]
    for _ in range(400):
      o = hc.update(25 * MPH, lead(d=30, v=25 * MPH), p, 1.0, STOCK_T[p], 1.2, 10.0, False, DT, plan_a_min=0.0)
      assert o.t_follow <= STOCK_T[p] + 1e-9
      assert prev - o.t_follow <= HC.DENSE_RATE_DOWN * DT + 1e-9
      prev = o.t_follow
    assert o.t_follow == pytest.approx(min(STOCK_T[p], HC.dense_t_target(25 * MPH, p)))


def test_no_dense_change_below_4mph_or_without_lead():
  hc = HC.HumanCity()
  o = run_hc(hc, 200, 3.5 * MPH, lead(d=8, v=1.5))
  assert o.t_follow == pytest.approx(1.45)
  hc = HC.HumanCity()
  run_hc(hc, 200, 25 * MPH, lead(d=30, v=25 * MPH))
  o = run_hc(hc, 1, 25 * MPH, lead(present=False))
  assert o.t_follow == pytest.approx(1.45) and o.dense == 0.0


@pytest.mark.parametrize("bad", [
  dict(a=-1.0),                    # lead braking
  dict(v=25 * MPH - 2.0),          # closing > 1.5 m/s
  dict(d=5.0, v=25 * MPH - 1.0),   # TTC < 6 s
  dict(d=float("nan")), dict(v=float("nan")), dict(a=float("nan")),
])
def test_calm_gate_fails_closed_and_t_returns_promptly(bad):
  hc = HC.HumanCity()
  run_hc(hc, 300, 25 * MPH, lead(d=30, v=25 * MPH))
  off = hc.out.dense
  assert off > 0.3
  ld = lead(**{**dict(d=30, v=25 * MPH, a=0.0), **bad})
  o = run_hc(hc, int(math.ceil(off / (HC.DENSE_RATE_UP * DT))) + 1, 25 * MPH, ld)
  assert o.t_follow == pytest.approx(1.45)


def test_fcw_and_urgency_restore_stock():
  for kw in (dict(fcw=True), dict(plan_a_min=-2.0)):
    hc = HC.HumanCity()
    run_hc(hc, 300, 25 * MPH, lead(d=30, v=25 * MPH))
    o = run_hc(hc, 20, 25 * MPH, lead(d=30, v=25 * MPH), **kw)
    assert o.t_follow == pytest.approx(1.45)
    assert o.jerk_factor == pytest.approx(1.0) or o.urgency > 0


def test_jerk_factor_and_danger_bounds():
  hc = HC.HumanCity()
  for v, ld, pa in [(10.0, lead(v=2.0), 0.0), (20.0, lead(v=20.0, a=-3.0), -3.0), (25.0, lead(present=False), 0.0)]:
    o = hc.update(v, ld, "aggressive", 0.5, 1.25, 1.5, 10.0, False, DT, plan_a_min=pa)
    assert o.jerk_factor >= 0.5 - 1e-9
    assert o.danger_factor is None or o.danger_factor >= HC.LEAD_DANGER_FACTOR_STOCK
    assert o.danger_cost >= HC.DANGER_ZONE_COST_STOCK
    assert o.cap <= 1.5 + 1e-9  # city default never raises the accel ceiling
  # Aggressive calm city: Relaxed-style braking floor
  o = run_hc(HC.HumanCity(), 2, 10.0, lead(d=20, v=10.0), pers="aggressive", base_t=1.25)
  assert o.jerk_factor >= HC.RELAXED_JERK_FLOOR


def test_boost_only_for_slow_or_braking_lead():
  assert HC.boost_weight(lead(v=25 * MPH, a=0.0)) == 0.0          # similar-speed cut-in: stock weighting
  assert HC.boost_weight(lead(v=1.0, a=0.0)) == 1.0               # slow / stopped lead
  assert HC.boost_weight(lead(v=20.0, a=-1.0)) == 1.0             # braking lead
  assert HC.boost_weight(lead(v=float("nan"))) == 1.0             # fail careful
  assert HC.boost_weight(lead(present=False)) == 0.0


def test_disabled_passthrough():
  hc = HC.HumanCity(HC.HumanCityConfig(enabled=False))
  o = hc.update(10.0, lead(), "standard", 1.0, 1.45, 1.2, 10.0, False, DT)
  assert o.t_follow is None and o.jerk_factor is None and o.danger_cost is None and o.cap == 1.2


# --- lead hold-over ---------------------------------------------------------------

def test_holdover_holds_briefly_and_only_adds_obstacle():
  h = HC.LeadHoldover()
  real = lead(d=10.0, v=3.0, a=-1.0)
  assert h.update(real, 4.0, DT) is real
  held = [h.update(lead(present=False), 4.0, DT) for _ in range(int(HC.HOLD_S / DT) + 3)]
  live = [x for x in held if getattr(x, "held", False)]
  assert 0 < len(live) <= int(HC.HOLD_S / DT) + 1
  assert all(x.modelProb == 0.0 for x in live)            # never raises FCW by itself
  vs = [x.vLead for x in live]
  assert all(b <= a + 1e-9 for a, b in zip(vs, vs[1:]))   # never predicted to speed up
  assert all(x.aLeadK <= 0.0 for x in live)
  assert not getattr(held[-1], "held", False)             # expires
  # reappears -> real lead immediately
  h.update(lead(present=False), 4.0, DT)
  r2 = lead(d=9.0, v=2.0)
  assert h.update(r2, 4.0, DT) is r2


@pytest.mark.parametrize("last", [lead(d=50.0, v=3.0), lead(d=10.0, v=8.0), lead(d=float("nan"))])
def test_holdover_not_for_far_pulling_away_or_bad_lead(last):
  h = HC.LeadHoldover()
  h.update(last, 4.0, DT)
  assert not getattr(h.update(lead(present=False), 4.0, DT), "held", False)


# --- launch take-up ceiling (LAUNCH CHANGE; human_launch) ---------------------------

def test_human_launch_cap_gated():
  hc = HC.HumanCity(HC.HumanCityConfig(human_launch=True))
  go = lead(d=12.0, v=2.0, a=1.0)
  o = hc.update(1.0, go, "standard", 1.0, 1.45, 1.25, 10.0, False, DT, plan_a_min=0.0)
  assert o.cap == pytest.approx(max(1.25, HC.human_launch_cap(1.0, "standard")))
  for bad in (lead(d=12.0, v=2.0, a=-0.5), lead(d=12.0, v=0.0, a=0.0), lead(d=9.0, v=2.0, a=1.0),
              lead(d=12.0, v=0.5, a=1.0), lead(present=False), lead(d=float("nan"), v=2.0, a=1.0)):
    o = hc.update(1.0, bad, "standard", 1.0, 1.45, 1.25, 10.0, False, DT, plan_a_min=0.0)
    assert o.cap == pytest.approx(1.25), bad
  o = hc.update(1.0, go, "standard", 1.0, 1.45, 1.25, 10.0, True, DT, plan_a_min=0.0)  # FCW
  assert o.cap == pytest.approx(1.25)
  ordering = [HC.human_launch_cap(0.0, p) for p in PERS]
  assert ordering[0] > ordering[1] > ordering[2]
  assert HC.human_launch_cap(30 * MPH, "aggressive") == 0.0


# --- source tripwires ---------------------------------------------------------------

def test_planner_wiring():
  src = (PKG / "selfdrive/controls/lib/longitudinal_planner.py").read_text()
  assert "self.human_city.update(" in src and "self.lead_holdover.update(" in src
  assert "self.mpc.update(radar_state," in src and "t_follow=t_follow" in src
  assert "jerk_factor=jerk_factor, danger_cost=danger_cost" in src
  mpc = (PKG / "selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py").read_text()
  assert "STOP_DISTANCE = 6.0" in mpc
  assert "max(DANGER_ZONE_COST, float(danger_cost))" in mpc
  assert "max(LEAD_DANGER_FACTOR, float(self.lead_danger_factor))" in mpc


def test_panda_and_safety_doc_untouched():
  r = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "9c6e321c06", "HEAD", "--", "panda", "docs/SAFETY.md",
                      "opendbc_repo/opendbc/safety"], capture_output=True, text=True)
  if r.returncode != 0:
    pytest.skip("git history not available")
  assert r.stdout.strip() == ""


# --- closed loop --------------------------------------------------------------------

CITY = {"name": "city", "human": {"human_launch": False}, "release": False}
LAUNCH = {"name": "launch", "human": {"human_launch": True}, "release": True}
STOCK = {"name": "dev-c4", "human": None, "release": False}


def _sim(cand, pers):
  build = Path(os.environ.get("TEST3_MPC_BUILD", "/tmp/test3dev_mpc_build"))
  build.mkdir(parents=True, exist_ok=True)
  out = Path(tempfile.mkdtemp()) / "human.json"
  env = dict(os.environ)
  extra = [str(ROOT), str(ROOT / "opendbc_repo"), str(ROOT / "msgq_repo")]
  env["PYTHONPATH"] = os.pathsep.join([p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p] + extra)
  subprocess.run([sys.executable, str(HERE / "human_city_sim.py"), str(ROOT), str(build), str(out), json.dumps(cand), pers],
                 check=True, env=env, cwd=tempfile.gettempdir(), capture_output=True, timeout=1500)
  return json.loads(out.read_text())["cases"][pers]


def _has_release():
  return importlib.util.find_spec("opendbc.sunnypilot.car.toyota.launch_release") is not None


@pytest.fixture(scope="module")
def sims():
  if importlib.util.find_spec("acados") is None:
    pytest.skip("host acados build not available")
  cands = [("city", CITY), ("stock", STOCK)] + ([("launch", LAUNCH)] if _has_release() else [])
  jobs = {(c, p): (cand, p) for c, cand in cands for p in PERS}
  with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {k: ex.submit(_sim, *v) for k, v in jobs.items()}
    return {k: f.result() for k, f in futs.items()}


def _builds(sims):
  return [b for b in ("city", "launch") if (b, "aggressive") in sims]


@pytest.mark.parametrize("p", PERS)
def test_zero_collisions_and_no_solver_failures(sims, p):
  for b in _builds(sims):
    for name, r in sims[(b, p)].items():
      assert not r["collision"], (b, p, name)
      assert r["solve_fail"] == 0, (b, p, name)


@pytest.mark.parametrize("p", PERS)
def test_near_miss_family_brakes_and_holds_set_gap(sims, p):
  for b in _builds(sims):
    for name, r in sims[(b, p)].items():
      if r["kind"] != "nearmiss":
        continue
      # ego always came to a stop behind the lead on its own (no driver input in sim)
      assert r["final_gap_m"] > 0.0
      if r["set_gap_check"]:
        # never meaningfully closer than Matt's set stopping distance
        assert r["min_gap_minus_stop_gap_m"] >= -0.30, (b, p, name, r["min_gap_minus_stop_gap_m"])
      stock = sims[("stock", p)][name]
      assert r["min_gap_m"] >= stock["min_gap_m"] - 0.05, (b, p, name)


@pytest.mark.parametrize("p", PERS)
def test_ordinary_stops_hold_set_gap(sims, p):
  for b in _builds(sims):
    for name in ("signal_queue_40mph", "signal_queue_35mph_firm", "approach_stopped_queue_40mph",
                 "approach_stopped_queue_45mph", "lead_slows_to_stop_40mph", "lead_hard_brake_40mph_3.5",
                 "accordion_0_15mph", "stopped_lead_hold", "launch_to35mph"):
      r = sims[(b, p)][name]
      assert r["min_gap_minus_stop_gap_m"] >= -0.40, (b, p, name, r["min_gap_minus_stop_gap_m"])


@pytest.mark.parametrize("p", PERS)
def test_never_closer_than_stock_except_dense_cut_in(sims, p):
  for b in _builds(sims):
    for name, r in sims[(b, p)].items():
      stock = sims[("stock", p)][name]
      if r["kind"] == "headway":
        continue
      slack = 1.0 if name == "cutin_30mph_10m" else 0.05
      assert r["min_gap_m"] >= stock["min_gap_m"] - slack, (b, p, name, r["min_gap_m"], stock["min_gap_m"])


def test_beyond_limit_hard_brake_no_worse_than_stock(sims):
  for b in _builds(sims):
    for p in PERS:
      r, s = sims[(b, p)]["lead_hard_brake_40mph_4.5"], sims[("stock", p)]["lead_hard_brake_40mph_4.5"]
      assert r["min_gap_m"] >= s["min_gap_m"] - 0.05
      assert not r["collision"]


def test_stopped_lead_never_creeps(sims):
  for b in _builds(sims):
    for p in PERS:
      r = sims[(b, p)]["stopped_lead_hold"]
      assert r["v_ego_peak_mph"] < 0.1 and r["min_gap_minus_stop_gap_m"] >= -0.01


def test_city_headways_human_band_and_ordering(sims):
  for b in _builds(sims):
    for mph in (15, 25, 35, 45):
      hw = [sims[(b, p)][f"follow_{mph}mph"]["headway_s"] for p in PERS]
      st = [sims[("stock", p)][f"follow_{mph}mph"]["headway_s"] for p in PERS]
      assert hw[0] < hw[1] < hw[2], (b, mph, hw)
      assert all(h <= s + 1e-3 for h, s in zip(hw, st))
      assert min(hw) >= 1.30
    std = [sims[(b, "standard")][f"follow_{m}mph"]["headway_s"] for m in (25, 35, 45)]
    assert all(1.40 <= h <= 1.70 for h in std), std   # HCM discharge / Zhu 2016 medium-speed band


def test_city_build_adds_no_accel_and_no_hesitation(sims):
  for p in PERS:
    for name, r in sims[("city", p)].items():
      s = sims[("stock", p)][name]
      if r["kind"] == "headway":
        assert r["peak_accel"] <= 0.15, (p, name)  # gentle close-up to the dense gap
        continue
      assert r["peak_accel"] <= s["peak_accel"] + 0.1, (p, name)
      if r["t_move_mean"] is not None and s["t_move_mean"] is not None:
        assert r["t_move_mean"] <= s["t_move_mean"] + 0.08, (p, name)


def test_launch_build_human_start_and_accel(sims):
  if ("launch", "aggressive") not in sims:
    pytest.skip("launch_release not in this tree")
  for p in PERS:
    for name in ("signal_queue_40mph", "launch_to35mph", "accordion_0_15mph"):
      r = sims[("launch", p)][name]
      assert r["t_move_mean"] <= 1.15, (p, name)       # human ~1.04-1.08 s
      assert r["peak_accel"] <= HC.human_launch_cap(0.0, p) + 0.05


# --- LAUNCH CHANGE branch (dev-c4-human-launch) -------------------------------------
def test_launch_branch_defaults_human_launch_on():
  assert HC.HumanCityConfig().human_launch is True


def test_release_prev_accel_only_releases_standstill_hold():
  from opendbc.sunnypilot.car.toyota.launch_release import release_prev_accel
  # standstill, pid, positive request: deep hold -> light hold
  assert release_prev_accel(-2.0, 0.5, 0.0, True, True) == pytest.approx(-0.3)
  # braking request untouched
  assert release_prev_accel(-2.0, -0.5, 0.0, True, True) == -2.0
  # moving: untouched
  assert release_prev_accel(-2.0, 0.5, 3.0, True, True) == -2.0
  # still in stopping state (not pid): untouched
  assert release_prev_accel(-2.0, 0.5, 0.0, False, True) == -2.0
  # not long active: untouched
  assert release_prev_accel(-2.0, 0.5, 0.0, True, False) == -2.0
