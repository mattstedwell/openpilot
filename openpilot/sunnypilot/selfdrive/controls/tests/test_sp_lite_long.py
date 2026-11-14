"""dev-c4-sunnypilot-lite longitudinal changes (host tests; NOT road-tested).

1. +10 ft standstill gap behind every lead (long_mpc LEAD_STOP_EXTRA).
2. ~10 ft earlier stop for the driving model's light / stop-sign stops
   (lite_e2e_stop).
3. Quicker start after the lead pulls away (lite_launch + Toyota brake-hold
   release).
The closed-loop MPC checks need the host acados build (sp_lite_host_mpc)
and are skipped if acados is not importable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3]          # openpilot/ (python package dir)
TREE = PKG.parent              # worktree root
FT = 0.3048
BASE = "927767d51cdd8935e3402f45af1e024516692960"

from openpilot.sunnypilot.selfdrive.controls.lib import lite_launch as LL  # noqa: E402
from openpilot.sunnypilot.selfdrive.controls.lib import lite_e2e_stop as ES  # noqa: E402
from opendbc.sunnypilot.car.toyota.launch_release import release_prev_accel, LAUNCH_RELEASE_HOLD  # noqa: E402


def lead(d=9.3, v=0.6, a=0.3, present=True):
  return SimpleNamespace(present=present, dRel=d, vLead=v, aLeadK=a)


class TestLeadStopExtra(unittest.TestCase):
  def test_values(self):
    src = (PKG / "selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py").read_text()
    self.assertIn("LEAD_STOP_EXTRA = 10 * 0.3048", src)
    self.assertIn("radarstate.leadOne.present", src)
    self.assertIn("lead_0_obstacle = lead_0_obstacle - get_lead_stop_extra(lead_xv_0[:,1])", src)
    # compiled OCP untouched: STOP_DISTANCE is still official
    self.assertIn("STOP_DISTANCE = 6.0", src)

  def test_function(self):
    from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc as LM
    self.assertAlmostEqual(float(LM.get_lead_stop_extra(0.0)), 10 * FT, places=4)
    self.assertAlmostEqual(float(LM.get_lead_stop_extra(0.5)), 10 * FT, places=4)
    self.assertAlmostEqual(float(LM.get_lead_stop_extra(5.0)), 0.0, places=6)
    self.assertAlmostEqual(float(LM.get_lead_stop_extra(30.0)), 0.0, places=6)
    arr = LM.get_lead_stop_extra(np.array([0.0, 2.75, 10.0]))
    self.assertEqual(arr.shape, (3,))
    self.assertTrue(np.all(np.diff(arr) <= 0))


class TestLaunchRelease(unittest.TestCase):
  def test_release_only_standstill_pid_positive(self):
    self.assertEqual(release_prev_accel(-2.0, 0.5, 0.0, True, True), LAUNCH_RELEASE_HOLD)
    self.assertEqual(release_prev_accel(-2.0, -0.1, 0.0, True, True), -2.0)  # braking untouched
    self.assertEqual(release_prev_accel(-2.0, 0.5, 0.0, False, True), -2.0)  # still Stopping
    self.assertEqual(release_prev_accel(-2.0, 0.5, 0.5, True, True), -2.0)   # moving
    self.assertEqual(release_prev_accel(-2.0, 0.5, 0.0, True, False), -2.0)  # not active
    self.assertLess(LAUNCH_RELEASE_HOLD, 0.0)  # still a brake request

  def test_carcontroller_hook(self):
    src = (TREE / "opendbc_repo/opendbc/car/toyota/carcontroller.py").read_text()
    i = src.index("release_prev_accel(self.prev_accel")
    self.assertLess(i, src.index("pcm_accel_cmd = rate_limit(pcm_accel_cmd, self.prev_accel"))


class TestLiteLaunchGates(unittest.TestCase):
  def setUp(self):
    LL.ENABLED = True
    self.l = LL.LiteLaunch()
    # both stopped at the lite stop target: arms and sets d_ref
    for _ in range(5):
      self.l.update(lead(d=9.05, v=0.0, a=0.0), 0.0, True, False, 0.0)

  def go(self, ld, v_ego=0.0, fcw=False, a=0.05, a_e2e=None, ss_e2e=False, n=3):
    out = None
    for _ in range(n):
      out = self.l.update(ld, v_ego, v_ego < 1e-3, fcw, a, a_e2e, ss_e2e)
    return out

  def test_pulling_away_quick_start_and_floor(self):
    q, a = self.go(lead(d=9.4, v=0.6, a=0.5))
    self.assertTrue(q)
    self.assertAlmostEqual(a, LL.LAUNCH_FLOOR)

  def test_debounce(self):
    l = LL.LiteLaunch()
    l.update(lead(d=9.05, v=0.0, a=0.0), 0.0, True, False, 0.0)
    q, _ = l.update(lead(d=9.4, v=0.6, a=0.5), 0.0, True, False, 0.05)
    self.assertFalse(q)  # first cycle only

  def test_never_lowers_plan(self):
    _, a = self.go(lead(d=9.4, v=0.6, a=0.5), a=1.2)
    self.assertEqual(a, 1.2)

  def test_blocks(self):
    cases = {
      "lead slowing": dict(ld=lead(d=9.4, v=0.6, a=-0.2)),
      "lead slow": dict(ld=lead(d=9.4, v=0.2, a=0.3)),
      "lead not moved": dict(ld=lead(d=9.1, v=0.6, a=0.3)),
      "gap inside target": dict(ld=lead(d=8.9, v=0.6, a=0.3)),
      "lead lost": dict(ld=lead(present=False)),
      "fcw": dict(ld=lead(d=9.4, v=0.6, a=0.3), fcw=True),
      "plan braking": dict(ld=lead(d=9.4, v=0.6, a=0.3), a=-0.5),
      "model stop": dict(ld=lead(d=9.4, v=0.6, a=0.3), ss_e2e=True),
      "not faster than ego": dict(ld=lead(d=9.4, v=0.6, a=0.3), v_ego=0.5),
    }
    for name, kw in cases.items():
      with self.subTest(name):
        self.setUp()
        q, a = self.go(**kw)
        self.assertFalse(q, name)
        self.assertEqual(a, kw.get("a", 0.05), name)

  def test_ends_above_launch_speed_and_disarms(self):
    self.go(lead(d=9.4, v=0.6, a=0.5))
    q, _ = self.go(lead(d=12.0, v=3.0, a=0.5), v_ego=LL.LAUNCH_V_MAX + 0.1)
    self.assertFalse(q)
    q, _ = self.go(lead(d=12.0, v=3.0, a=0.5), v_ego=1.0)
    self.assertFalse(q)  # disarmed until the next standstill

  def test_floor_capped_by_model(self):
    q, a = self.go(lead(d=9.4, v=0.6, a=0.5), a=0.05, a_e2e=0.1)
    self.assertTrue(q)
    self.assertAlmostEqual(a, 0.1)

  def test_unlatched_gate_drops_immediately(self):
    self.go(lead(d=9.4, v=0.6, a=0.5))
    q, a = self.l.update(lead(d=9.4, v=0.6, a=-0.3), 0.05, False, False, -0.1)
    self.assertFalse(q)
    self.assertEqual(a, -0.1)

  def test_disabled(self):
    LL.ENABLED = False
    try:
      self.assertEqual(self.go(lead(d=9.4, v=0.6, a=0.5)), (False, 0.05))
    finally:
      LL.ENABLED = True

  def test_planner_wiring(self):
    src = (PKG / "selfdrive/controls/lib/longitudinal_planner.py").read_text()
    i_min = src.index("output_a_target, self.mpc.source, _ = min(candidates")
    i_ll = src.index("self.lite_launch.update(")
    i_clip = src.index("self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)")
    self.assertLess(i_min, i_ll)
    self.assertLess(i_ll, i_clip)
    i_shift = src.index("shift_e2e_stop(sm['modelV2']")
    self.assertLess(i_shift, src.index("candidates = [("))


T = np.array([10 * (i / 32) ** 2 for i in range(33)])  # ModelConstants.T_IDXS


def model_plan(x_ego, v_ego, line, go=False, a_brake=1.2, braking=False):
  """Synthetic model: brake at a_brake to stop at `line`, hold; or accelerate (go)."""
  if go:
    a = 1.0
    v = v_ego + a * T
    x = v_ego * T + 0.5 * a * T ** 2
    return SimpleNamespace(position=SimpleNamespace(x=list(x), t=list(T)), velocity=SimpleNamespace(x=list(v))), a, False
  d = max(line - x_ego, 0.01)
  need = v_ego ** 2 / (2 * d)
  a = -need if (need >= a_brake or braking) else 0.0
  v = np.maximum(v_ego + a * T, 0.0)
  x = np.minimum(v_ego * T + 0.5 * a * T ** 2, d) if a < 0 else v_ego * T
  if a < 0:
    t_stop = v_ego / -a
    x = np.where(T >= t_stop, d, x)
  ss = v_ego < 0.3 and d < 1.0
  return SimpleNamespace(position=SimpleNamespace(x=list(x), t=list(T)), velocity=SimpleNamespace(x=list(v))), a, ss


def run_light(lite, v0=40 * 0.44704, line=120.0, lead_obj=None, t_go=None):
  ES.ENABLED = lite
  x, v, a_act, dt = 0.0, v0, 0.0, 0.05
  peak = 0.0
  stopped_x = None
  t = 0.0
  launched = False
  braking = False
  while t < 60.0:
    go = t_go is not None and t >= t_go
    m, a_m, ss_m = model_plan(x, v, line, go=go, braking=braking)
    braking = braking or a_m < 0
    a_cmd, ss = ES.shift_e2e_stop(m, lead_obj, v, a_m, ss_m)
    if ss or (v < 0.3 and a_cmd < 0.1):  # drive_helpers.should_stop -> longcontrol Stopping
      a_cmd = min(a_cmd, -1.0)
    a_act += (a_cmd - a_act) * (1 - np.exp(-dt / 0.25))
    v = max(0.0, v + a_act * dt)
    x += v * dt
    peak = min(peak, a_act)
    if v < 0.01 and stopped_x is None and t > 1.0:
      stopped_x = x
    if go and v > 1.0:
      launched = True
      break
    t += dt
  ES.ENABLED = True
  return stopped_x, peak, launched


class TestE2eStopShift(unittest.TestCase):
  def test_stops_about_10ft_earlier(self):
    for mph in (25, 35, 45):
      with self.subTest(mph=mph):
        xs, ps, _ = run_light(False, v0=mph * 0.44704)
        xl, pl, _ = run_light(True, v0=mph * 0.44704)
        self.assertAlmostEqual(xs - xl, 10 * FT, delta=0.6)
        self.assertGreaterEqual(pl, min(ps, -ES.MAX_EXTRA_DECEL) - 0.05)  # no harsher than model or the 2.0 cap

  def test_green_launches(self):
    xl, _, launched = run_light(True, t_go=40.0)
    self.assertIsNotNone(xl)
    self.assertTrue(launched)

  def test_lead_explains_stop_passthrough(self):
    m, a, ss = model_plan(0.0, 10.0, 40.0)
    self.assertEqual(ES.shift_e2e_stop(m, lead(d=41.0, v=0.0, a=0.0), 10.0, a, ss), (a, ss))

  def test_no_stop_passthrough(self):
    m, a, ss = model_plan(0.0, 10.0, 0.0, go=True)
    self.assertEqual(ES.shift_e2e_stop(m, None, 10.0, a, ss), (a, ss))

  def test_never_adds_accel_random(self):
    rng = np.random.default_rng(0)
    for _ in range(300):
      v = float(rng.uniform(0, 25)); line = float(rng.uniform(0.5, 150)); a_in = float(rng.uniform(-3, 2))
      m, _, ss = model_plan(0.0, v, line, go=bool(rng.random() < 0.3))
      a_out, ss_out = ES.shift_e2e_stop(m, None, v, a_in, ss)
      self.assertLessEqual(a_out, a_in + 1e-9)
      self.assertTrue(ss_out or not ss)

  def test_bad_model_passthrough(self):
    m = SimpleNamespace(position=SimpleNamespace(x=[], t=[]), velocity=SimpleNamespace(x=[]))
    self.assertEqual(ES.shift_e2e_stop(m, None, 5.0, -0.5, False), (-0.5, False))


def rlead(d=9.3, v=0.6, a=0.3, present=True):
  # radarState lead incl. aLeadTau (the holdover needs all four values)
  return SimpleNamespace(present=present, dRel=d, vLead=v, aLeadK=a, aLeadTau=1.5)


class TestLeadHoldover(unittest.TestCase):
  def setUp(self):
    from openpilot.sunnypilot.selfdrive.controls.lib import lite_lead_holdover as LH
    self.LH = LH
    LH.ENABLED = True
    self.h = LH.LeadHoldover()

  def test_real_lead_passthrough(self):
    ld = rlead(d=20.0, v=5.0, a=-1.0)
    self.assertIs(self.h.update(ld, 6.0, 0.05), ld)

  def test_held_lead_decelerates_never_accelerates_no_fcw(self):
    self.h.update(rlead(d=15.0, v=6.0, a=-1.5), 6.5, 0.05)
    prev_v = 6.0
    for _ in range(20):  # 1.0 s
      h = self.h.update(rlead(present=False), 6.0, 0.05)
      self.assertTrue(getattr(h, "held", False))
      self.assertLessEqual(h.vLead, prev_v)
      self.assertLessEqual(h.aLeadK, 0.0)
      self.assertEqual(h.modelProb, 0.0)
      prev_v = h.vLead
    self.assertLess(prev_v, 6.0 - 2.9)  # 2x the last -1.5 decel for 1 s
    self.assertFalse(getattr(self.h.update(rlead(present=False), 6.0, 0.05), "held", False))  # > 1 s: released

  def test_positive_last_accel_held_at_zero(self):
    self.h.update(rlead(d=15.0, v=6.0, a=1.0), 6.0, 0.05)
    h = self.h.update(rlead(present=False), 6.0, 0.05)
    self.assertEqual(h.aLeadK, 0.0)
    self.assertAlmostEqual(h.vLead, 6.0)

  def test_no_hold_far_or_pulling_away(self):
    self.h.update(rlead(d=45.0, v=6.0, a=0.0), 6.0, 0.05)
    self.assertFalse(getattr(self.h.update(rlead(present=False), 6.0, 0.05), "held", False))
    self.h.update(rlead(d=15.0, v=9.0, a=0.0), 6.0, 0.05)
    self.assertFalse(getattr(self.h.update(rlead(present=False), 6.0, 0.05), "held", False))

  def test_planner_wiring_mpc_only(self):
    src = (PKG / "selfdrive/controls/lib/longitudinal_planner.py").read_text()
    i = src.index("self.lead_holdover.update(radar_state.leadOne")
    self.assertLess(i, src.index("self.mpc.update(radar_state"))
    # launch and e2e-stop gates still see the real radar lead
    self.assertIn("self.lite_launch.update(sm['radarState'].leadOne", src)
    self.assertIn("shift_e2e_stop(sm['modelV2'], sm['radarState'].leadOne", src)
    mpc_src = (PKG / "selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py").read_text()
    self.assertIn("radarstate.leadOne.modelProb > 0.9", mpc_src)  # held modelProb 0.0 -> no FCW


class TestSafetyUntouched(unittest.TestCase):
  def test_safety_paths_unchanged_vs_upstream(self):
    try:
      out = subprocess.run(["git", "-C", str(TREE), "diff", "--name-only", BASE, "--", "panda", "docs/SAFETY.md",
                            "opendbc_repo/opendbc/safety", "launch_env.sh"], capture_output=True, text=True, timeout=60)
    except Exception as e:  # pragma: no cover
      self.skipTest(f"git unavailable: {e}")
    if out.returncode != 0:
      self.skipTest(out.stderr)
    self.assertEqual(out.stdout.strip(), "")


def _acados_ok():
  try:
    import acados  # noqa: F401
    return True
  except Exception:
    return False


@unittest.skipUnless(_acados_ok(), "host acados not available")
class TestClosedLoopMpc(unittest.TestCase):
  """Runs sp_lite_launch_sim.py (real MPC) on the near-miss + roll-stop family, Aggressive."""
  CASES = "nm_,rollstop_,launch_,stopped_lead_hold,signal_queue_40mph,lead_hard_brake_25mph,lead_hard_brake_40mph_3.5"

  @classmethod
  def setUpClass(cls):
    build = os.environ.get("SP_LITE_MPC_BUILD", "/tmp/sp_lite_mpc_build")
    cls.res = {}
    with tempfile.TemporaryDirectory() as d:
      for name, cand in (("stock", {"name": "stock"}),
                         ("lite", {"name": "lite", "extra": True, "release": True, "launch": True, "holdover": True})):
        out = Path(d) / f"{name}.json"
        env = dict(os.environ)
        subprocess.run([sys.executable, str(HERE / "sp_lite_launch_sim.py"), str(TREE), build, str(out),
                        json.dumps(cand), "aggressive", cls.CASES], check=True, capture_output=True, env=env,
                       timeout=900)
        cls.res[name] = json.loads(out.read_text())["cases"]["aggressive"]

  def test_lite_no_new_collisions(self):
    for n, r in self.res["lite"].items():
      with self.subTest(n):
        self.assertFalse(r["collision"])  # incl. nm_lead_lost_while_stopping_15mph (stock collides)

  def test_standstill_gap_plus_10ft(self):
    for n in ("stopped_lead_hold", "signal_queue_40mph"):
      s, l = self.res["stock"][n], self.res["lite"][n]
      self.assertAlmostEqual(l["min_gap_m"] - s["min_gap_m"], 10 * FT, delta=0.15)

  def test_rollstop_gaps_do_not_shrink(self):
    for n, r in self.res["lite"].items():
      if not n.startswith("rollstop_"):
        continue
      with self.subTest(n):
        g = r["hold_end_gap_m"][1:]  # after the first roll
        self.assertGreaterEqual(min(g), 6.0 + 1.0)  # well above the stock 6.0 m stop distance
        self.assertGreaterEqual(min(g), min(self.res["stock"][n]["hold_end_gap_m"][1:]) + 2.5)
        half = len(g) // 2
        self.assertGreaterEqual(min(g[half:]), min(g[:half]) - 0.25)  # no downward trend

  def test_quicker_start(self):
    for n in ("launch_to35mph", "launch_gentle_0.5", "launch_brisk_2.0", "nm_creep_3m_stop"):
      with self.subTest(n):
        self.assertLess(self.res["lite"][n]["t_move"][0], self.res["stock"][n]["t_move"][0] - 0.2)

  def test_no_solver_failures(self):
    for name in ("stock", "lite"):
      for n, r in self.res[name].items():
        self.assertEqual(r["solve_fail"], 0, f"{name} {n}")


if __name__ == "__main__":
  unittest.main()
