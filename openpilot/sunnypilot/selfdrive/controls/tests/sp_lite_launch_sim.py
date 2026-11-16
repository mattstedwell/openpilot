#!/usr/bin/env python3
"""dev-c4-sunnypilot-lite closed-loop host sim: stock sunnypilot dev vs lite.

usage: sp_lite_launch_sim.py TREE_ROOT BUILD_DIR OUT_JSON CAND_JSON [personalities] [name-substrings]
CAND_JSON: {"name": str, "extra": bool, "release": bool, "launch": bool}
  extra   -> long_mpc LEAD_STOP_EXTRA (+10 ft standstill gap); False sets it to 0.0 (stock)
  release -> Toyota brake-hold release (opendbc sunnypilot launch_release)
  launch  -> lite_launch planner gates
  holdover-> lite_lead_holdover (phantom lead for <= 1 s radar dropouts) (quick start after lead moves)

Real LongitudinalMpc (host acados build of this tree), the planner's
MPC + cruise path wired like longitudinal_planner.py, real LongControl, and
the Toyota TSS2 carcontroller long path emulated (4 m/s^3 rate limit, TSS2
PID, a_ego_future). Radar: aLeadK lags the true accel by 0.3 s, aLeadTau
follows radard. Plant: 0.05 s delay + 0.25 s lag. NOT simulated: the
driving model's e2e desiredAcceleration / shouldStop candidate (treated as
non-limiting), radard Kalman lag on vLead, the Toyota PCM's own resume
delay from standstill, road grade. Host-only; NOT road-tested.
"""
import json, math, sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sp_lite_host_mpc as HM  # noqa: E402

TREE = Path(sys.argv[1]).resolve()
BUILD = Path(sys.argv[2])
OUT = Path(sys.argv[3])
CAND = json.loads(sys.argv[4])
HM.setup(BUILD, TREE)

import numpy as np  # noqa: E402
from openpilot.cereal import log  # noqa: E402
from opendbc.car.structs import car  # noqa: E402
import openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc as LM  # noqa: E402
import openpilot.selfdrive.controls.lib.longitudinal_planner as LP  # noqa: E402
import openpilot.selfdrive.controls.lib.longcontrol as LCmod  # noqa: E402
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop  # noqa: E402
from openpilot.selfdrive.modeld.constants import ModelConstants  # noqa: E402
from opendbc.car.common.pid import PIDController  # noqa: E402
from opendbc.car.common.filter_simple import FirstOrderFilter  # noqa: E402
from opendbc.car import rate_limit  # noqa: E402
try:
  from opendbc.sunnypilot.car.toyota.launch_release import release_prev_accel
except Exception:
  release_prev_accel = None
try:
  from openpilot.sunnypilot.selfdrive.controls.lib.lite_lead_holdover import LeadHoldover
except Exception:
  LeadHoldover = None
try:
  from openpilot.sunnypilot.selfdrive.controls.lib import lite_launch as LL
except Exception:
  LL = None

import os
TRACE = [float(x) for x in os.environ['SIM_TRACE'].split(',')] if os.environ.get('SIM_TRACE') else None
if not CAND.get("extra", False):
  LM.LEAD_STOP_EXTRA = 0.0
  if hasattr(LM, "STOP_TAPER_COMP"):
    LM.STOP_TAPER_COMP = 0.0
for _k, _v in CAND.get("lm", {}).items():  # exploratory overrides only
  setattr(LM, _k, _v)
try:
  import openpilot.sunnypilot.selfdrive.controls.lib.lite_fusion as LFmod  # noqa: E402
except Exception:
  LFmod = None
for _k, _v in CAND.get("lf", {}).items():  # exploratory overrides only
  setattr(LFmod, _k, _v)
for _k, _v in CAND.get("lc", {}).items():  # exploratory overrides only
  setattr(LCmod, _k, _v)
USE_LAUNCH = bool(CAND.get("launch", False) and LL is not None)
if LL is not None:
  LL.ENABLED = USE_LAUNCH
  for _k, _v in CAND.get("ll", {}).items():  # exploratory overrides only
    setattr(LL, _k, _v)
try:
  from openpilot.sunnypilot.selfdrive.controls.lib import lite_launch_boost as LB
except Exception:
  LB = None
USE_BOOST = bool(CAND.get("boost", False) and LB is not None)
if LB is not None:
  LB.ENABLED = USE_BOOST
  for _k, _v in CAND.get("lb", {}).items():  # exploratory overrides only
    setattr(LB, _k, _v)
try:
  from openpilot.sunnypilot.selfdrive.controls.lib import lite_lead_lock as LK
except Exception:
  LK = None
USE_LOCK = bool(CAND.get("lock", False))
try:
  from openpilot.sunnypilot.selfdrive.controls.lib import lite_turn_cap as TC
except Exception:
  TC = None
if TC is not None:
  TC.ENABLED = bool(CAND.get("turncap", False))
try:
  from openpilot.sunnypilot.selfdrive.controls.lib import lite_gap_open as GO
except Exception:
  GO = None
USE_GAPOPEN = bool(CAND.get("gapopen", False) and GO is not None)
if GO is not None:
  GO.ENABLED = USE_GAPOPEN
  for _k, _v in CAND.get("go", {}).items():
    setattr(GO, _k, _v)
if LK is not None:
  LK.ENABLED = USE_LOCK
  for _k, _v in CAND.get("lk", {}).items():
    setattr(LK, _k, _v)
try:
  from openpilot.sunnypilot.selfdrive.controls.lib import lite_e2e_stop as ES
except Exception:
  ES = None
if ES is not None:
  for _k, _v in CAND.get("es", {}).items():
    setattr(ES, _k, _v)
# "e2e": include an emulated driving-model e2e candidate (desiredAcceleration /
# shouldStop + a plan for lite_e2e_stop) in the planner min, like the real
# planner. Scenarios without a "model" spec get a free-road model (non-limiting).
USE_E2E = bool(CAND.get("e2e", False))
BOOST_CLEARS_SS = bool(CAND.get("boost_clears_ss", False))  # planner of the fix branch clears shouldStop when boosting
MODEL_SMOOTH = float(CAND.get("model_smooth", 0.3))  # stock modeld LONG_SMOOTH_SECONDS
MODEL_LAG = float(CAND.get("model_lag", 0.4))  # s: how late the emulated model sees the lead move ("lead" spec)
if CAND.get("rel_hold") is not None:
  import opendbc.sunnypilot.car.toyota.launch_release as _LR
  _LR.LAUNCH_RELEASE_HOLD = float(CAND["rel_hold"])

P = log.LongitudinalPersonality
PNAME = {P.aggressive: "aggressive", P.standard: "standard", P.relaxed: "relaxed"}
LongCtrlState = car.CarControl.Actuators.LongControlState
CTRL_T = ModelConstants.T_IDXS[:CONTROL_N]
ACCEL_MIN, ACCEL_MAX = -3.5, 2.0
MPH = 0.44704
LOGROWS = []


def stop_target():
  return LM.STOP_DISTANCE + float(LM.LEAD_STOP_EXTRA)


def eq_gap(p, v):
  return float(LM.get_T_FOLLOW(p) * v + LM.STOP_DISTANCE)


def S(name, segs, ev0=0.0, lv0=0.0, gap0=None, v_cruise=45 * MPH, cut=None, t_tail=10.0, noise=False,
      lost=None, appear=None, kind="city", headway_mph=None, set_gap_check=True, model=None, still=None,
      t_ref=None, flicker=None, vision=None):
  # model: list of (t, state, arg) for the emulated driving model (needs CAND e2e):
  #   ("stop", L) plan a stop at absolute x=L (red light / stop sign line); ("go", A) pull away at A
  # still: (t0, t1) window where ego must not move (displacement reported)
  # t_ref: time from which t5/t10/t20 are measured when there is no lead start (e.g. green)
  # flicker: (t0, t1, on_s, off_s) radar lead present/absent flicker
  return dict(name=name, segs=segs, ev0=ev0, lv0=lv0, gap0=gap0, v_cruise=v_cruise, cut=cut, t_tail=t_tail,
              noise=noise, lost=lost or [], appear=appear, kind=kind, headway_mph=headway_mph,
              set_gap_check=set_gap_check, model=model, still=still, t_ref=t_ref, flicker=flicker,
              vision=vision, gone_after=None, steer=None)


def accordion(peaks, ups, dns, wait=4.0, hold=2.0):
  segs = [("hold", 2.0)]
  for vp, au, ad in zip(peaks, ups, dns):
    segs += [("acc", au, vp * MPH), ("hold", hold), ("acc", ad, 0.0), ("hold", wait)]
  return segs


def signal_cycle(v_peak, n=3, up=1.5, dn=-1.8, green=12.0, red=20.0):
  segs = [("hold", 3.0)]
  for _ in range(n):
    segs += [("acc", up, v_peak * MPH), ("hold", green), ("acc", dn, 0.0), ("hold", red)]
  return segs


def roll_stop(n, D, au, ad, hold):
  return [("hold", 3.0)] + sum([[("dist", au, ad, D), ("hold", hold)] for _ in range(n)], [])


SCENARIOS = [
  *[S(f"follow_{m}mph", [("hold", 45.0)], ev0=m * MPH, lv0=m * MPH, kind="headway", headway_mph=m, t_tail=0.0,
      v_cruise=(m + 5) * MPH) for m in (15, 25, 35, 45)],
  S("signal_queue_40mph", signal_cycle(40, up=1.5, dn=-1.8)),
  S("signal_queue_35mph_firm", signal_cycle(35, up=2.0, dn=-2.5)),
  S("approach_stopped_queue_40mph", [("hold", 1.0)], ev0=40 * MPH, lv0=0.0, gap0=120.0, t_tail=25.0),
  S("approach_stopped_queue_45mph", [("hold", 1.0)], ev0=45 * MPH, lv0=0.0, gap0=140.0, t_tail=25.0),
  S("lead_slows_to_stop_40mph", [("hold", 4.0), ("acc", -1.5, 0.0), ("hold", 8.0)], ev0=40 * MPH, lv0=40 * MPH),
  S("lead_hard_brake_40mph_3.5", [("hold", 4.0), ("acc", -3.5, 0.0), ("hold", 8.0)], ev0=40 * MPH, lv0=40 * MPH),
  S("lead_hard_brake_40mph_4.5", [("hold", 4.0), ("acc", -4.5, 0.0), ("hold", 8.0)], ev0=40 * MPH, lv0=40 * MPH,
    kind="beyond_limit"),
  S("lead_hard_brake_25mph", [("hold", 4.0), ("acc", -4.0, 0.0), ("hold", 8.0)], ev0=25 * MPH, lv0=25 * MPH),
  S("accordion_0_15mph", accordion([8, 15, 12, 5, 15], [1.0, 1.5, 1.2, 0.8, 2.0], [-1.5, -2.5, -2.0, -1.5, -3.0])),
  S("accordion_noisy_radar", accordion([10, 15, 6, 12], [1.2, 1.5, 0.8, 1.0], [-2.0, -2.5, -1.5, -3.0]), noise=True),
  S("rolling_5_25mph", [("hold", 3.0)] + sum([[("acc", 1.0, 25 * MPH), ("hold", 2.0), ("acc", -1.5, 5 * MPH), ("hold", 1.0)]
                                              for _ in range(4)], []), ev0=5 * MPH, lv0=5 * MPH),
  S("launch_to35mph", [("hold", 2.0), ("acc", 1.5, 35 * MPH), ("hold", 10.0)], kind="launch"),
  S("launch_gentle_0.5", [("hold", 2.0), ("acc", 0.5, 20 * MPH), ("hold", 6.0)], kind="launch"),
  S("launch_brisk_2.0", [("hold", 2.0), ("acc", 2.0, 30 * MPH), ("hold", 6.0)], kind="launch"),
  S("stopped_lead_hold", [("hold", 15.0)]),
  S("cutin_30mph_10m", [("hold", 12.0)], ev0=30 * MPH, lv0=30 * MPH, cut=dict(t=3.0, gap=10.0, v=28 * MPH),
    set_gap_check=False),
  S("cutin_12mph_then_stop", [("hold", 4.5), ("acc", -1.5, 0.0), ("hold", 6.0)], ev0=12 * MPH, lv0=12 * MPH,
    cut=dict(t=3.0, gap=7.0, v=10 * MPH), set_gap_check=False),
  # Nov 13 near-miss family
  S("nm_creep_1.5m_stop", [("hold", 2.0), ("dist", 1.0, -1.5, 1.5), ("hold", 6.0)], kind="nearmiss"),
  S("nm_creep_3m_stop", [("hold", 2.0), ("dist", 1.0, -1.5, 3.0), ("hold", 6.0)], kind="nearmiss"),
  S("nm_creep_3m_slow_up", [("hold", 2.0), ("dist", 0.5, -1.5, 3.0), ("hold", 6.0)], kind="nearmiss"),
  S("nm_creep_then_hard_brake", [("hold", 2.0), ("dist", 1.0, -3.5, 3.0), ("hold", 6.0)], kind="nearmiss"),
  S("nm_start_then_stop_4mph", [("hold", 2.0), ("acc", 1.0, 4 * MPH), ("acc", -4.0, 0.0), ("hold", 6.0)], kind="nearmiss"),
  S("nm_start_then_stop_8mph", [("hold", 2.0), ("acc", 1.5, 8 * MPH), ("acc", -4.0, 0.0), ("hold", 6.0)], kind="nearmiss"),
  S("nm_lead_lost_during_creep", [("hold", 2.0), ("dist", 1.0, -2.0, 3.0), ("hold", 6.0)], lost=[(3.0, 4.0)],
    kind="nearmiss"),
  S("nm_lead_lost_while_stopping_15mph", [("hold", 3.0), ("acc", -3.0, 0.0), ("hold", 8.0)], ev0=15 * MPH,
    lv0=15 * MPH, lost=[(3.2, 3.8)], kind="nearmiss"),
  S("nm_lead_appears_late_stopped_30mph", [("hold", 1.0)], ev0=30 * MPH, lv0=0.0, gap0=55.0, appear=0.0, t_tail=10.0,
    kind="nearmiss"),
  S("nm_lead_appears_late_crawl_10mph", [("hold", 1.0), ("acc", -1.0, 0.0), ("hold", 6.0)], ev0=10 * MPH,
    lv0=3 * MPH, gap0=16.0, appear=0.0, kind="nearmiss"),
  S("nm_cutin_crawl_close", [("hold", 3.0), ("acc", -2.0, 0.0), ("hold", 6.0)], ev0=5 * MPH, lv0=5 * MPH,
    cut=dict(t=2.0, gap=4.0, v=4 * MPH), kind="nearmiss", set_gap_check=False),
  S("nm_cutin_crawl_8m_stop", [("hold", 2.5), ("acc", -2.5, 0.0), ("hold", 6.0)], ev0=8 * MPH, lv0=8 * MPH,
    cut=dict(t=2.0, gap=8.0, v=6 * MPH), kind="nearmiss", set_gap_check=False),
  # Matt's real-world failure: roll, stop, roll, stop ... gap must not shrink cycle to cycle
  S("rollstop_1m_x8", roll_stop(8, 1.0, 1.0, -1.5, 1.5), kind="rollstop"),
  S("rollstop_2m_x8", roll_stop(8, 2.0, 1.0, -2.0, 1.5), kind="rollstop"),
  S("rollstop_3m_x8", roll_stop(8, 3.0, 1.2, -2.5, 2.0), kind="rollstop"),
  S("rollstop_2m_short_hold_x8", roll_stop(8, 2.0, 1.5, -3.0, 0.5), kind="rollstop"),
  S("rollstop_0.5m_x10", roll_stop(10, 0.5, 0.8, -1.5, 1.0), kind="rollstop"),
  S("rollstop_5m_x6", roll_stop(6, 5.0, 1.0, -2.0, 3.0), kind="rollstop"),
  S("rollstop_2m_noisy_x8", roll_stop(8, 2.0, 1.0, -2.0, 1.5), noise=True, kind="rollstop"),
  # --- launchboost (Matt t372u/t374u + "remove the delay from a stop" + "ensure it's still safe")
  *[S(f"lp_lead_{a}", [("hold", 2.0), ("acc", a, 40 * MPH), ("hold", 6.0)], kind="launchperf")
    for a in (0.5, 1.0, 1.5, 2.0, 2.5)],
  # lead pulls away, then brakes to a stop again (incl. beyond-limit -4.5)
  *[S(f"adv_go_then_brake_{d}_{m}mph", [("hold", 2.0), ("acc", 2.0, m * MPH), ("acc", -d, 0.0), ("hold", 6.0)],
      kind="adversarial") for d in (3.0, 3.5, 4.5) for m in (5, 10, 20)],
  # creeping lead, many short cycles (0.5-1 m)
  *[S(f"adv_creep_{d}m_x8", roll_stop(8, d, 0.8, -1.5, 1.0), kind="rollstop") for d in (0.5, 0.75, 1.0)],
  S("adv_creep_0.75m_noisy_x8", roll_stop(8, 0.75, 0.8, -1.5, 1.0), noise=True, kind="rollstop"),
  # radar drops the lead during the launch (and it is braking when it returns)
  S("adv_lost_during_launch", [("hold", 2.0), ("acc", 2.0, 12 * MPH), ("acc", -3.0, 0.0), ("hold", 6.0)],
    lost=[(3.2, 3.9)], kind="adversarial"),
  S("adv_lost_during_launch_late", [("hold", 2.0), ("acc", 1.5, 15 * MPH), ("acc", -3.5, 0.0), ("hold", 6.0)],
    lost=[(5.0, 5.9)], kind="adversarial"),
  # cut-in from the adjacent lane while we are launching (slower car, short gap), then it stops
  *[S(f"adv_cutin_launch_{m}mph", [("hold", 2.0), ("acc", 2.0, 25 * MPH), ("hold", 1.0), ("acc", -3.0, 0.0), ("hold", 6.0)],
      cut=dict(t=tc, gap=g, v=max(m - 3, 1) * MPH), kind="adversarial", set_gap_check=False)
    for m, tc, g in ((5, 3.6, 5.0), (10, 4.6, 7.0), (15, 5.6, 9.0))],
  # flickering / noisy radar lead at standstill: must not creep
  S("adv_flicker_stopped_lead", [("hold", 15.0)], noise=True, flicker=(1.0, 15.0, 0.25, 0.15), still=(0.0, 15.0),
    kind="adversarial"),
  S("adv_flicker_then_launch", [("hold", 4.0), ("acc", 1.5, 20 * MPH), ("hold", 4.0)], noise=True,
    flicker=(1.0, 6.0, 0.3, 0.1), kind="adversarial"),
]

NOLEAD = dict(lost=[(0.0, 1e9)], gap0=500.0)
LIGHT_L = 40.0  # m: stop line ahead of the start point for the model scenarios
MODEL_SCENARIOS = [
  # red light from 25 mph, green at 14 s (model-driven stop, no lead)
  S("ml_red_then_green", [("hold", 30.0)], ev0=25 * MPH, model=[(0.0, "stop", LIGHT_L), (14.0, "go", 1.5)],
    t_ref=14.0, still=(9.0, 14.0), kind="model", **NOLEAD),
  S("ml_red_then_green_gentle", [("hold", 30.0)], ev0=25 * MPH, model=[(0.0, "stop", LIGHT_L), (14.0, "go", 0.6)],
    t_ref=14.0, still=(9.0, 14.0), kind="model", **NOLEAD),
  # stop sign: full stop, then the model plans to go 2 s after stopping
  S("ml_stop_sign", [("hold", 30.0)], ev0=20 * MPH, model=[(0.0, "stop", LIGHT_L), (13.0, "go", 1.2)],
    t_ref=13.0, still=(10.5, 13.0), kind="model", **NOLEAD),
  # model stop prediction flickers at a red light: must not creep
  S("ml_red_flicker", [("hold", 30.0)], ev0=25 * MPH,
    model=[(0.0, "stop", LIGHT_L)] + sum([[(12.0 + 0.6 * i, "go", 1.0), (12.25 + 0.6 * i, "stop", LIGHT_L)] for i in range(15)], []),
    still=(9.0, 22.0), kind="model", **NOLEAD),
  # green, then a pedestrian / stopped car is revealed ahead (radar), model does not see it
  S("ml_green_then_stopped_obstacle", [("hold", 20.0)], ev0=25 * MPH,
    model=[(0.0, "stop", LIGHT_L), (14.0, "go", 1.5)], t_ref=14.0, gap0=LIGHT_L - 3.05 + 9.0, appear=14.8,
    kind="model"),
  # stop sign, then cross traffic shows up as a ~stationary radar lead just ahead for 2 s
  S("ml_stop_sign_cross_traffic", [("hold", 20.0)], ev0=20 * MPH,
    model=[(0.0, "stop", LIGHT_L), (13.0, "go", 1.2)], t_ref=13.0, gap0=LIGHT_L - 3.05 + 7.0, appear=13.3,
    lost=[(15.3, 1e9)], kind="model"),
]
MODEL_SCENARIOS[-1]["gone_after"] = 15.3   # the crossing car has left the lane
for _sc in MODEL_SCENARIOS:
  if _sc["lost"] and _sc["lost"][0][0] == 0.0:
    _sc["gone_after"] = 0.0                  # no physical lead at all
# vision + radar fusion cases (model lead emulated with 0.2 s lag unless overridden)
SCENARIOS += [
  # radar drops the stopped lead at standstill (common low-speed Toyota dropout): must not creep
  S("fus_radar_dropout_standstill", [("hold", 15.0)], lost=[(2.0, 2.6), (5.0, 6.5), (9.0, 9.4)], still=(0.0, 15.0),
    kind="fusion", vision=("fallback", 0.0, 1e9)),
  # same dropouts while the lead creeps and stops (radar gone, vision only, radard-style fallback)
  S("fus_radar_dropout_creep", [("hold", 2.0), ("dist", 0.8, -1.5, 1.0), ("hold", 2.0), ("dist", 0.8, -1.5, 1.0), ("hold", 6.0)],
    lost=[(2.2, 3.4), (5.5, 6.8)], kind="fusion", vision=("fallback", 0.0, 1e9)),
  # radar says the lead pulls away but the model still sees it stopped for 1 s: no early launch
  S("fus_vision_disagrees", [("hold", 2.0), ("acc", 1.5, 15 * MPH), ("hold", 4.0)], vision=("stopped", 2.0, 3.0),
    kind="fusion"),
  # model lead flickers (prob in/out) during a launch
  S("fus_vision_flicker_launch", [("hold", 2.0), ("acc", 2.0, 20 * MPH), ("acc", -3.0, 0.0), ("hold", 5.0)],
    vision=("flicker", 1.0, 8.0), kind="fusion"),
  # vision-only lead (no radar track) pulling away: stock launch only
  S("fus_vision_only_launch", [("hold", 2.0), ("acc", 1.5, 15 * MPH), ("hold", 4.0)], vision=("only", 0.0, 1e9),
    kind="fusion"),
]
SCENARIOS += MODEL_SCENARIOS
# realistic lead launches (sunnypilot-lite-test-fix, Matt's road report: car did not move / weak launch):
# the emulated driving model plans to stop m_gap behind the lead and only pulls away once it SEES the lead
# moving (MODEL_LAG late), like the real e2e candidate. Needs CAND e2e; t_ref is the lead start.
RL_SCENARIOS = []
for _mg in (3.0, 4.5, 6.0):
  for _a in (1.0, 1.5, 2.5):
    RL_SCENARIOS.append(S(f"rl_stop_launch_g{_mg}_a{_a}", [("hold", 12.0), ("acc", _a, 30 * MPH), ("hold", 8.0)],
                          ev0=25 * MPH, lv0=0.0, gap0=60.0, model=[(0.0, "lead", (_mg, 1.5))], kind="rl_launch", t_ref=12.0))
# radar lead flickers / is vision-only at standstill and during the departure
RL_SCENARIOS += [
  S("rl_flicker_launch", [("hold", 12.0), ("acc", 1.5, 30 * MPH), ("hold", 8.0)], ev0=25 * MPH, gap0=60.0,
    model=[(0.0, "lead", (4.5, 1.5))], flicker=(9.0, 14.0, 0.3, 0.2), kind="rl_launch", t_ref=12.0),
  S("rl_radar_absent_launch", [("hold", 12.0), ("acc", 1.5, 30 * MPH), ("hold", 8.0)], ev0=25 * MPH, gap0=60.0,
    model=[(0.0, "lead", (4.5, 1.5))], lost=[(10.0, 13.5)], vision=("fallback", 0.0, 1e9), kind="rl_launch", t_ref=12.0),
  S("rl_vision_only_launch", [("hold", 12.0), ("acc", 1.5, 30 * MPH), ("hold", 8.0)], ev0=25 * MPH, gap0=60.0,
    model=[(0.0, "lead", (4.5, 1.5))], vision=("only", 9.0, 1e9), kind="rl_launch", t_ref=12.0),
  # lead pulls away then stops again after a short roll: must not inch closer
  S("rl_roll_stop_x6", [("hold", 12.0)] + sum([[("dist", 1.0, -2.0, 2.0), ("hold", 1.5)] for _ in range(6)], []),
    ev0=25 * MPH, gap0=60.0, model=[(0.0, "lead", (4.5, 1.5))], kind="rollstop"),
  # lead goes, then brakes hard after the launch
  S("rl_go_then_brake", [("hold", 12.0), ("acc", 2.0, 12 * MPH), ("acc", -4.0, 0.0), ("hold", 6.0)], ev0=25 * MPH,
    gap0=60.0, model=[(0.0, "lead", (4.5, 1.5))], kind="adversarial"),
]
SCENARIOS += RL_SCENARIOS
# stop-and-go lead lock (Matt, Nov 14): all after a real approach to a stopped lead, realistic model
_LL = dict(ev0=25 * MPH, gap0=60.0, model=[(0.0, "lead", (4.5, 1.5))], t_ref=12.0)
LL_SCENARIOS = [
  S("ll_hard_launch", [("hold", 12.0), ("acc", 3.0, 25 * MPH), ("hold", 6.0)], kind="lock", **_LL),
  S("ll_slow_launch", [("hold", 12.0), ("acc", 0.5, 15 * MPH), ("hold", 6.0)], kind="lock", **_LL),
  S("ll_creep_stop_x8", [("hold", 12.0)] + sum([[("dist", 0.6, -1.0, 0.7), ("hold", 1.0)] for _ in range(8)], []),
    kind="rollstop", **_LL),
  S("ll_go_brake_3mph", [("hold", 12.0), ("acc", 2.0, 3 * MPH), ("acc", -4.0, 0.0), ("hold", 6.0)], kind="lock", **_LL),
  S("ll_radar_dropout", [("hold", 12.0), ("acc", 1.5, 20 * MPH), ("hold", 6.0)], lost=[(12.6, 13.4)], kind="lock", **_LL),
  S("ll_cutin", [("hold", 12.0), ("acc", 1.0, 10 * MPH), ("hold", 8.0)], cut=dict(t=13.5, gap=4.0, v=0.5),
    set_gap_check=False, kind="lock", **_LL),
  S("ll_rollback", [("hold", 12.0), ("back", 0.3, 1.0), ("hold", 4.0), ("acc", 1.5, 15 * MPH), ("hold", 6.0)],
    still=(12.0, 17.0), kind="lock", **{k: v for k, v in _LL.items() if k != "t_ref"}),
  S("ll_noisy_stopped_hold", [("hold", 25.0)], noise=True, still=(10.0, 25.0), kind="lock",
    **{k: v for k, v in _LL.items() if k != "t_ref"}),
]
SCENARIOS += LL_SCENARIOS
# low-speed 90 deg right turn (Matt): following a lead at 12 mph, set speed 35 mph, wheel -300 deg for 6 s;
# the lead turns out of view mid-turn (radar lost, physically gone): the car must not accelerate hard in the corner
for _v in (10, 12, 15):
  _sc = S(f"turn_right_lead_lost_{_v}mph", [("hold", 30.0)], ev0=_v * MPH, lv0=_v * MPH, v_cruise=35 * MPH,
          lost=[(5.5, 1e9)], kind="turn", set_gap_check=False, t_tail=0.0)
  _sc["steer"] = [(4.0, 9.0, -300.0)]
  _sc["gone_after"] = 5.5
  SCENARIOS.append(_sc)


class Lead:
  def __init__(self, segs, v0):
    self.segs = list(segs); self.v = v0; self.a = 0.0; self.i = 0; self.t_seg = 0.0
    self._peak = False; self._peak_i = -1

  def step(self, t, dt):
    a = 0.0
    while self.i < len(self.segs):
      seg = self.segs[self.i]
      if seg[0] == "hold":
        if self.t_seg < seg[1]:
          break
      elif seg[0] == "acc":
        _, acc, vt = seg
        if (acc > 0 and self.v < vt - 1e-6) or (acc < 0 and self.v > vt + 1e-6):
          a = acc; break
      elif seg[0] == "back":  # ("back", speed, duration): lead rolls backward (negative speed)
        if self.t_seg < seg[2]:
          self.a = 0.0; self.v = -abs(seg[1]); self.t_seg += dt
          return self.v
        self.v = 0.0
      elif seg[0] == "dist":
        _, au, ad, D = seg
        vp = math.sqrt(2 * D / (1 / au + 1 / abs(ad)))
        if self._peak_i != self.i:
          self._peak = False; self._peak_i = self.i
        if not self._peak and self.v < vp:
          a = au; break
        self._peak = True
        if self.v > 1e-6:
          a = ad; break
      self.i += 1; self.t_seg = 0.0
    self.a = a
    v_new = self.v + a * dt
    seg = self.segs[self.i] if self.i < len(self.segs) else None
    if seg is not None and seg[0] == "acc":
      v_new = min(v_new, seg[2]) if seg[1] > 0 else max(v_new, seg[2])
    self.v = max(v_new, 0.0)
    self.t_seg += dt
    return self.v

  def done(self):
    return self.i >= len(self.segs)


class ModelEmu:
  """Driving-model e2e stand-in: raw accel + a plan (for lite_e2e_stop), modeld-style smoothing."""
  T = np.array(ModelConstants.T_IDXS)

  def __init__(self, spec, v_cruise):
    self.spec = spec or []; self.vc = v_cruise; self.a_s = 0.0; self.t_go = None; self.st = "free"
    self.a_raw = 2.0

  def _plan_acc(self, ev, a):
    v = np.minimum(np.maximum(ev + a * self.T, 0.0), max(self.vc, ev))
    x = np.concatenate([[0.0], np.cumsum(np.diff(self.T) * (v[1:] + v[:-1]) / 2.0)])
    return x, v

  def update(self, t, ex, ev, lead_obs=None):
    st, arg = "free", None
    for te, s_, a_ in self.spec:
      if t >= te - 1e-9:
        st, arg = s_, a_
    if st == "lead":
      # behind a lead: the model plans to stop model_gap behind the (lagged) lead
      # while it still sees the lead stopped, and pulls away (ramp to a_go) once it
      # sees it moving. lead_obs = (gap, v_lead, physical) as perceived (lagged).
      m_gap, a_go = arg
      g, lvo, phys = lead_obs if lead_obs is not None else (1e3, 30.0, False)
      if phys and lvo < 0.3:
        st, arg = "stop", ex + max(g - m_gap, 0.0)
      else:
        st, arg = "go", a_go
    if st == "go" and self.st != "go":
      self.t_go = t
    self.st = st
    if st == "free":
      a_raw = 2.0; x, v = self._plan_acc(ev, a_raw)
    elif st == "go":
      a_raw = arg * (1.0 - math.exp(-(t - self.t_go) / 0.5)); x, v = self._plan_acc(ev, a_raw)
    else:
      d = arg - ex
      if ev > 0.3:
        a_raw = -ev * ev / (2.0 * max(d, 0.3))
        v = np.maximum(ev + a_raw * self.T, 0.0)
        x = np.minimum(np.concatenate([[0.0], np.cumsum(np.diff(self.T) * (v[1:] + v[:-1]) / 2.0)]), max(d, 0.0))
      elif d > 1.0:  # creep up to the line, then stay
        a_raw = 0.3
        x = d * (1.0 - np.exp(-self.T / 2.0)); v = (d / 2.0) * np.exp(-self.T / 2.0)
      else:
        a_raw = -0.2; x = np.zeros_like(self.T); v = np.zeros_like(self.T)
    self.a_raw = a_raw
    alpha = 1.0 - math.exp(-0.05 / MODEL_SMOOTH) if MODEL_SMOOTH > 0 else 1.0
    self.a_s = self.a_s + alpha * (a_raw - self.a_s)
    ss_m = bool(ev < 0.3 and a_raw < 0.1)
    return SimpleNamespace(position=SimpleNamespace(x=list(x), t=list(self.T)), velocity=SimpleNamespace(x=list(v)),
                           action=SimpleNamespace(desiredAcceleration=float(self.a_s), shouldStop=ss_m))


def LINE_OF(sc):
  for _, s_, a_ in (sc["model"] or []):
    if s_ == "stop":
      return a_
  return None


def _call_launch(launch, *args, model_lead=None, **kw):
  try:
    return launch.update(*args, model_lead=model_lead, **kw)
  except TypeError:
    try:
      return launch.update(*args, model_lead=model_lead)
    except TypeError:
      return launch.update(*args)


def steer_at(sc, t):
  """(wheel angle deg, rate deg/s) for sc["steer"] = [(t0, t1, angle)], 1.5 s ramps in/out."""
  for t0_, t1_, a_ in sc.get("steer") or []:
    if t0_ <= t < t1_:
      r = 1.5
      if t < t0_ + r:
        return a_ * (t - t0_) / r, a_ / r
      if t > t1_ - r:
        return a_ * (t1_ - t) / r, -a_ / r
      return a_, 0.0
  return 0.0, 0.0


def run(personality, sc):
  use_rel = bool(CAND.get("release", False) and release_prev_accel is not None)
  mpc = LM.LongitudinalMpc()
  es_state = ES.E2eStopState() if (ES is not None and hasattr(ES, 'E2eStopState')) else None
  g_stop = stop_target()
  ev = sc["ev0"]; lv = sc["lv0"]
  gap0 = sc["gap0"] if sc["gap0"] is not None else (g_stop if lv < 0.1 else eq_gap(personality, lv))
  lead = Lead(sc["segs"], lv)
  cut = sc["cut"]
  CP = SimpleNamespace(stopAccel=-2.0, vEgoStopping=0.25, vEgoStarting=0.25,
                       longitudinalTuning=SimpleNamespace(kiBP=[0.], kiV=[0.]))
  lc = LCmod.LongControl(CP, SimpleNamespace(enableGasInterceptor=False))
  launch = LL.LiteLaunch() if USE_LAUNCH else None
  boost = LB.LiteLaunchBoost() if USE_BOOST else None
  lock = LK.LeadLock() if (USE_LOCK and LK is not None) else None
  gapopen = GO.LiteGapOpen() if (USE_GAPOPEN and GO is not None) else None
  lock_on = 0; gapopen_on = 0; ego_a_onset = None; a_at = {}; gap_at = {}; turn_peak_a = 0.0; turn_peak_ea = 0.0; turn_g = {}
  model = ModelEmu(sc["model"], sc["v_cruise"]) if USE_E2E else None
  use_fusion = bool(CAND.get("fusion", True))
  vis_hist = []
  t_ref = sc["t_ref"]; t_clear = None; t_move_clear = None; t_move_ref = None
  tv = {}; t_launch0 = None
  still_x0 = None; still_disp = 0.0
  stop_hist = []; stop_comfort = []
  boost_on = 0
  hold = LeadHoldover() if (CAND.get("holdover", False) and LeadHoldover is not None) else None
  pid = PIDController(0.0, ([2., 5.], [0.5, 0.25]), k_f=1.0, pos_limit=2.0, neg_limit=-3.5, rate=1 / 0.03)
  aego_f = FirstOrderFilter(0.0, 0.25, 0.03)
  vdf = FirstOrderFilter(ev, 2.0, 0.05)
  v_cruise = sc["v_cruise"]
  dt_c = 0.01
  ex, ea = 0.0, 0.0
  lx = gap0; la_f = 0.0; tau_f = 1.5
  out_a = 0.0; plan_a = 0.0; ss = ev < 0.3; a_cruise = 0.0; prev_accel = 0.0; sent = 0.0
  quick = False
  delay = [0.0] * int(sc.get("plant_delay_n", 5))
  p_tau = float(sc.get("plant_tau", 0.25)); b_tau = float(sc.get("brake_release_tau") or p_tau)

  def visible(t):
    if sc["appear"] is not None and t < sc["appear"] + 1e-9 and sc["appear"] > 0:
      return False
    fl = sc["flicker"]
    if fl is not None and fl[0] <= t < fl[1] and ((t - fl[0]) % (fl[2] + fl[3])) >= fl[2]:
      return False
    return not any(a <= t < b for a, b in sc["lost"])

  def physical(t):
    if sc["appear"] is not None and sc["appear"] > 0 and t < sc["appear"]:
      return False
    return sc.get("gone_after") is None or t < sc["gone_after"]
  if ev > 0.1 and sc["appear"] is None:
    for _ in range(40):
      mpc.set_weights(True, personality=personality)
      mpc.set_cur_state(ev, 0.0)
      mpc.update(HM.radar(gap0, lv, 0.0), personality=personality)
  cut_done = False; solve_fail = 0
  rng = np.random.default_rng(1234)
  t = 0.0; k = 0
  min_gap = 1e9; min_gap_stop_rel = 1e9; peak_a = 0.0; peak_d = 0.0; peak_j = 0.0; jerk_sq = 0.0; n_j = 0
  ea_hist = []; v_hist = []; collided = False
  firsts = []; t_first_pending = None
  moves = []; lead_was_stopped = lv < 0.05; t_lead_start = None
  hold_end = []
  stop_gaps = []; ego_stopped_prev = ev < 0.05
  hw_samples = []; t_end_lead = None; ego_moved_while_lead_stopped = 0.0
  while t < 300.0:
    lv = lead.step(t, dt_c)
    la_true = lead.a
    if cut and not cut_done and t >= cut["t"]:
      cut_done = True
      lx = ex + cut["gap"]; lv = cut["v"]; lead.v = lv; la_f = 0.0
    lx += lv * dt_c
    la_f += (la_true - la_f) * (1 - math.exp(-dt_c / 0.3))
    gap = lx - ex
    if lead.done() and t_end_lead is None:
      t_end_lead = t
    if t_end_lead is not None and t > t_end_lead + sc["t_tail"]:
      break
    if lv < 0.05:
      lead_was_stopped = True
    elif lead_was_stopped and lv > 0.05:
      lead_was_stopped = False
      if t > 0.5:
        hold_end.append((round(lx - ex, 2), round(ev / MPH, 2)))
      if ev < 0.05:
        t_lead_start = t
        if t_launch0 is None:
          t_launch0 = t
    if t_lead_start is not None and ev > 0.01 and t_first_pending is None:
      t_first_pending = t_lead_start
      firsts.append(round(t - t_lead_start, 3))
    if t_lead_start is not None and ev > 0.1:
      moves.append(round(t - t_lead_start, 3)); t_lead_start = None; t_first_pending = None
    if t_lead_start is not None and lv < 0.01 and ev < 0.01:
      t_lead_start = None; t_first_pending = None  # lead stopped again before we moved: not a launch sample
    if k % 5 == 0:
      if abs(la_f) < 0.5:
        tau_f = 1.5
      else:
        tau_f += (0.0 - tau_f) * (0.05 / (0.45 + 0.05))
      vis = visible(t)
      vis_hist.append((t, lv, gap, physical(t)))
      if sc["noise"]:
        rs = HM.radar(gap + rng.normal(0, 0.15), max(lv + rng.normal(0, 0.15), 0.0), la_f + rng.normal(0, 0.3),
                      tau=tau_f, present=vis)
      else:
        rs = HM.radar(gap, lv, la_f, tau=tau_f, present=vis)
      if vis:
        rs.leadOne.vRel = float(lv - ev)
        rs.leadOne.radarTrackId = 2 if cut_done else 1
      # emulated model lead (vision): 0.2 s behind the truth, prob 0.9 while the lead is physically there
      model_lead = None
      if use_fusion:
        old = [h for h in vis_hist if h[0] <= t - 0.2]
        h = old[-1] if old else vis_hist[0]
        prob = 0.9 if h[3] else 0.0
        mv = h[1]
        vs = sc["vision"]
        if vs is not None and vs[1] <= t < vs[2]:
          if vs[0] == "stopped":
            mv = 0.0
          elif vs[0] == "flicker":
            prob = prob if int((t - vs[1]) / 0.2) % 2 == 0 else 0.0
        model_lead = SimpleNamespace(prob=prob, x=[h[2] + 1.52], v=[mv], a=[0.0])
        if vs is not None and vs[0] == "fallback" and not rs.leadOne.present and prob > 0.5:
          rs = HM.radar(h[2], mv, 0.0, tau=0.3, present=True)
          rs.leadOne.radar = False
          rs.leadOne.vRel = float(mv - ev)
        if vs is not None and vs[0] == "only" and vs[1] <= t < vs[2] and rs.leadOne.present:
          rs.leadOne.radar = False
      standstill = ev < 1e-3
      vdf.x = max(0.0, vdf.update(ev))
      mpc.set_weights(not standstill, personality=personality)
      mpc.set_cur_state(vdf.x, out_a)
      rs_mpc = rs
      if hold is not None:
        held = hold.update(rs.leadOne, ev, 0.05)
        if held is not rs.leadOne:
          rs_mpc = SimpleNamespace(leadOne=held, leadTwo=rs.leadTwo)
      mpc.update(rs_mpc, personality=personality)
      if mpc.solution_status != 0:
        solve_fail += 1
      fcw = mpc.crash_cnt > 2 and not standstill
      vt = np.interp(CTRL_T, LM.T_IDXS, mpc.v_solution)
      at = np.interp(CTRL_T, LM.T_IDXS, mpc.a_solution)
      a_prev = out_a
      a_mpc = get_accel_from_plan(vt, at, CTRL_T, action_t=0.05 + 0.05)
      ss_mpc = should_stop(ev, a_mpc)
      a_cruise = LP.get_cruise_accel(v_cruise, ev, a_cruise, 0.05)
      ss_cruise = should_stop(ev, a_cruise)
      a_e2e = None; ss_e2e = False; model_stop_x = None
      if model is not None:
        _ml = [h for h in vis_hist if h[0] <= t - MODEL_LAG]
        _mh = _ml[-1] if _ml else vis_hist[0]
        m = model.update(t, ex, ev, lead_obs=(_mh[2], _mh[1], _mh[3]))
        model_stop_x = ES.planned_stop_x(m) if ES is not None else None
        a_e2e = m.action.desiredAcceleration; ss_e2e = m.action.shouldStop
        if ES is not None:
          if es_state is not None:
            a_e2e, ss_e2e = ES.shift_e2e_stop(m, rs.leadOne, ev, a_e2e, ss_e2e, state=es_state, dt=0.05)
          else:
            a_e2e, ss_e2e = ES.shift_e2e_stop(m, rs.leadOne, ev, a_e2e, ss_e2e)
        if t_ref is not None and t >= t_ref and t_clear is None and model.a_raw >= 0.1:
          t_clear = t
      a_t = float(np.clip(min(a_mpc, a_cruise, a_e2e if a_e2e is not None else 9.0), ACCEL_MIN, ACCEL_MAX))
      ss = ss_mpc or ss_cruise or ss_e2e
      if launch is not None:
        quick, a_t = _call_launch(launch, rs.leadOne, ev, standstill, fcw, a_t, a_e2e, ss_e2e, model_lead=model_lead,
                                  model_stop_x=model_stop_x, a_plan_radar=min(a_mpc, a_cruise), dt=0.05)
        if quick:
          ss = False
        elif getattr(launch, "pending", False):
          ss = False
          a_t = LL.PRE_HOLD if a_t <= 0.0 else a_t
      if boost is not None:
        on, a_t = _call_launch(boost, rs.leadOne, ev, standstill, fcw, personality, a_t, a_prev, 0.05, a_e2e, ss_e2e,
                               model_lead=model_lead, model_stop_x=model_stop_x, a_plan_radar=min(a_mpc, a_cruise))
        boost_on += int(on)
        if on and BOOST_CLEARS_SS:
          ss = False
      if lock is not None:
        lk_on, a_t = lock.update(rs.leadOne, ev, standstill, fcw, a_t, min(a_mpc, a_cruise),
                                 launch_active=bool(quick or (boost is not None and boost.active)),
                                 dt=0.05, stop_target=LL.stop_target() if LL is not None else None)
        lock_on += int(lk_on)
        if lk_on and a_t > 0.0:
          ss = False
      if gapopen is not None:
        go_on, a_t = gapopen.update(rs.leadOne, ev, standstill, fcw, personality, a_t, a_prev, 0.05,
                                    lock_active=bool(lock is not None and lock.active),
                                    boost_active=bool(boost is not None and boost.active))
        gapopen_on += int(go_on)
      if TC is not None and sc.get("steer"):
        ang, rate = steer_at(sc, t)
        a_t = float(TC.cap_accel(a_t, ev, TC.curvature_from_angle(ang), ang, rate))
        if abs(ang) > 30.0:
          turn_peak_a = max(turn_peak_a, a_t)
      out_a = a_t; plan_a = a_t
      if os.environ.get("SIM_LOGROWS"):
        LOGROWS.append(dict(t=round(t, 2), ev=ev, lv=lv, gap=gap, ea=ea,
                            a_mpc=float(a_mpc), a_cr=float(a_cruise),
                            a_e2e=(float(a_e2e) if a_e2e is not None else None),
                            a_out=float(a_t), quick=bool(quick),
                            boost=bool(boost.active) if boost is not None else False,
                            bfloor=float(boost.floor) if boost is not None else 0.0,
                            lockw=float(lock.weight) if lock is not None else 0.0,
                            locked=bool(lock.locked) if lock is not None else False,
                            gapopen=bool(gapopen.active) if gapopen is not None else False))
      vdf.x = vdf.x + 0.05 * (out_a + a_prev) / 2.0
    CS = SimpleNamespace(vEgo=ev, aEgo=ea, brakePressed=False, standstill=ev < 1e-3,
                         cruiseState=SimpleNamespace(standstill=False))
    act = float(lc.update(True, CS, plan_a, ss, (ACCEL_MIN, ACCEL_MAX)))
    if k % 3 == 0:
      if use_rel:
        prev_accel = release_prev_accel(prev_accel, act, ev, lc.long_control_state == LongCtrlState.pid, True)
      pcm = rate_limit(act, prev_accel, -4.0 * 0.03, 4.0 * 0.03)
      prev_accel = pcm
      prev_ae = aego_f.x; aego_f.update(ea)
      j_ego = (aego_f.x - prev_ae) / 0.03
      a_fut = ea + j_ego * float(np.interp(ev, [2., 5.], [0.25, 0.5]))
      pid.i -= 0.03 * 0.03 * float(np.sign(pid.i))
      pcm = pid.update(pcm - a_fut, speed=ev, feedforward=pcm,
                       freeze_integrator=lc.long_control_state != LongCtrlState.pid)
      sent = float(np.clip(pcm, -3.5, 2.0))
    if TRACE and TRACE[0] <= t <= TRACE[1] and k % 2 == 0:
      print(f"TR t={t:5.2f} lv={lv:5.2f} gap={lx-ex:6.2f} a_plan={plan_a:6.3f} ss={int(ss)} st={int(lc.long_control_state)} "
            f"act={act:6.3f} prev={prev_accel:6.3f} sent={sent:6.3f} ea={ea:6.3f} ev={ev:5.3f} q={int(quick)}")
    delay.append(sent); d = delay.pop(0)
    tau_now = b_tau if (ea < 0.0 and d > ea) else p_tau  # releasing the brake can be slower
    ea = ea + (d - ea) * (1 - math.exp(-dt_c / tau_now))
    ea_eff = 0.0 if (ev <= 0.0 and ea < 0.0) else ea
    ev = max(0.0, ev + ea_eff * dt_c)
    ex += ev * dt_c
    gap = lx - ex
    if lv < 0.01 and ev > 0.05 and gap < g_stop and physical(t):
      ego_moved_while_lead_stopped = max(ego_moved_while_lead_stopped, ev)
    t0v = t_ref if t_ref is not None else t_launch0
    if t0v is not None and t >= t0v:
      for mph in (5, 10, 20):
        if mph not in tv and ev >= mph * MPH:
          tv[mph] = round(t - t0v, 2)
      if t_move_ref is None and ev > 0.1:
        t_move_ref = round(t - t0v, 2)
    if t_clear is not None and t_move_clear is None and ev > 0.1:
      t_move_clear = round(t - t_clear, 2)
    if t_launch0 is not None and t >= t_launch0:
      if ego_a_onset is None and ea_eff > 0.05:
        ego_a_onset = round(t - t_launch0, 3)
      for dt_ in (0.5, 1.0, 2.0):
        if dt_ not in a_at and t >= t_launch0 + dt_:
          a_at[dt_] = round(ea_eff, 3)
      for dt_ in (3.0, 5.0):
        if dt_ not in gap_at and t >= t_launch0 + dt_:
          gap_at[dt_] = round(lx - ex, 2) if physical(t) else None
    if sc.get("steer"):
      ang_, _ = steer_at(sc, t)
      if abs(ang_) > 1.0:
        lat_ = ev * ev * abs(TC.curvature_from_angle(ang_)) if TC is not None else 0.0
        turn_peak_ea = max(turn_peak_ea, ea_eff)
        turn_g["lat"] = max(turn_g.get("lat", 0.0), lat_ / 9.81)
        turn_g["comb"] = max(turn_g.get("comb", 0.0), math.hypot(lat_, ea_eff) / 9.81)
        turn_g["v_exit"] = ev
    st_w = sc["still"]
    if st_w is not None and st_w[0] <= t < st_w[1]:
      if still_x0 is None:
        still_x0 = ex
      still_disp = max(still_disp, ex - still_x0)
    stop_hist.append((t, ea_eff, ev))
    if len(stop_hist) > 260:
      stop_hist.pop(0)
    if len(stop_hist) > 220 and ev < 0.01 and stop_hist[-2][2] >= 0.01:
      w = [h for h in stop_hist if h[0] >= t - 2.0]
      a_w = [h[1] for h in w]
      js = [abs(a_w[i] - a_w[i - 5]) / 0.05 for i in range(5, len(a_w) - 1)]  # rolling, excludes the stop instant
      prof = {f"{dt_:.1f}": round(float(np.interp(t - dt_, [h[0] for h in w], a_w)), 3) for dt_ in (2.0, 1.5, 1.0, 0.5, 0.2)}
      stop_comfort.append(dict(t=round(t, 2), gap=round(gap, 2) if physical(t) else None,
                               line_gap=round(LINE_OF(sc) - ex, 2) if LINE_OF(sc) is not None else None, peak_jerk_roll=round(max(js) if js else 0.0, 3),
                               a_at_stop=round(a_w[-2], 3), stop_step_jerk=round(abs(a_w[-2]) / 0.05, 2),
                               peak_decel=round(min(a_w), 3), prof=prof))
    if not (cut and not cut_done) and physical(t):
      min_gap = min(min_gap, gap)
      if lv < 0.05 and ev < 0.05:
        min_gap_stop_rel = min(min_gap_stop_rel, gap - g_stop)
    ego_stopped = ev < 0.01
    if ego_stopped and not ego_stopped_prev and lv < 0.05:
      stop_gaps.append(round(gap, 2))
    ego_stopped_prev = ego_stopped
    if gap <= 0.0 and physical(t):
      collided = True
    peak_a = max(peak_a, ea_eff); peak_d = min(peak_d, ea_eff)
    ea_hist.append(ea_eff); v_hist.append(ev)
    if len(v_hist) > 6:
      v_hist.pop(0)
    if len(ea_hist) > 5:
      j = (ea_hist[-1] - ea_hist[-6]) / 0.05
      ea_hist.pop(0)
      if ev > 0.1 and min(v_hist) > 0.1:
        peak_j = max(peak_j, abs(j)); jerk_sq += j * j; n_j += 1
    if sc["kind"] == "headway" and t > 35.0 and ev > 1.0:
      hw_samples.append(gap / ev)
    t += dt_c; k += 1
  return dict(collision=collided, min_gap_m=round(min_gap, 2), stop_target_m=round(g_stop, 2),
              min_gap_minus_target_m=round(min_gap - g_stop, 2),
              standstill_gap_minus_target_m=None if min_gap_stop_rel > 1e8 else round(min_gap_stop_rel, 2),
              stop_gaps_m=stop_gaps, hold_end_gap_m=[g for g, _ in hold_end], hold_end_v_mph=[v for _, v in hold_end], final_gap_m=round(lx - ex, 2), peak_accel=round(peak_a, 3),
              peak_decel=round(peak_d, 3), peak_jerk=round(peak_j, 2), rms_jerk=round(math.sqrt(jerk_sq / max(n_j, 1)), 3),
              t_first_motion=firsts, t_move=moves, t_move_mean=round(float(np.mean(moves)), 3) if moves else None,
              t_move_max=round(float(max(moves)), 3) if moves else None,
              headway_s=round(float(np.mean(hw_samples)), 3) if hw_samples else None,
              max_v_inside_target_lead_stopped=round(ego_moved_while_lead_stopped, 3),
              duration_s=round(t, 1), solve_fail=solve_fail, kind=sc["kind"], set_gap_check=sc["set_gap_check"],
              t_5mph=tv.get(5), t_10mph=tv.get(10), t_20mph=tv.get(20), t_move_ref=t_move_ref,
              t_clear=None if t_clear is None or t_ref is None else round(t_clear - t_ref, 2), t_move_after_clear=t_move_clear,
              still_disp_m=round(still_disp, 3) if sc["still"] is not None else None, stop_comfort=stop_comfort,
              boost_cycles=boost_on, lock_cycles=lock_on, gapopen_cycles=gapopen_on, t_ego_accel_onset=ego_a_onset,
              a_at_0_5=a_at.get(0.5), a_at_1_0=a_at.get(1.0), a_at_2_0=a_at.get(2.0),
              gap_at_3=gap_at.get(3.0), gap_at_5=gap_at.get(5.0),
              turn_peak_plan_a=round(turn_peak_a, 3) if sc.get("steer") else None,
              turn_peak_accel=round(turn_peak_ea, 3) if sc.get("steer") else None,
              turn_peak_lat_g=round(turn_g.get("lat", 0.0), 3) if sc.get("steer") else None,
              turn_peak_comb_g=round(turn_g.get("comb", 0.0), 3) if sc.get("steer") else None,
              turn_exit_mph=round(turn_g.get("v_exit", 0.0) / MPH, 1) if sc.get("steer") else None)


if os.environ.get("SIM_NO_MAIN"):
  raise SystemExit(0)
only = set(sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] else None
subs = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else None
res = {"tree": str(TREE), "candidate": CAND, "cases": {}}
for p in (P.aggressive, P.standard, P.relaxed):
  if only and PNAME[p] not in only:
    continue
  res["cases"][PNAME[p]] = {}
  for sc in SCENARIOS:
    if subs and not any(s in sc["name"] for s in subs):
      continue
    r = run(p, sc)
    res["cases"][PNAME[p]][sc["name"]] = r
    print(f"{CAND.get('name', '?'):6s} {PNAME[p]:10s} {sc['name']:36s} coll={int(r['collision'])} "
          f"min={r['min_gap_m']:6.2f}(tgt {r['stop_target_m']}) sgap-tgt={r['standstill_gap_minus_target_m']} "
          f"stops={r['stop_gaps_m'][:10]} hold_end={r['hold_end_gap_m'][:10]} hw={r['headway_s']} a+={r['peak_accel']:.2f} a-={r['peak_decel']:.2f} "
          f"jpk={r['peak_jerk']:.2f} t1={r['t_first_motion']} tmove={r['t_move']} fail={r['solve_fail']} "
          f"t5/10/20={r['t_5mph']}/{r['t_10mph']}/{r['t_20mph']} tclr={r['t_clear']} tmvclr={r['t_move_after_clear']} "
          f"still={r['still_disp_m']} boost={r['boost_cycles']}", flush=True)
OUT.write_text(json.dumps(res, indent=1))
