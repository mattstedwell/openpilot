"""sunnypilot-lite-test: smooth final stop, smoother light/sign stop, vision+radar
launch agreement, launch boost (off) — unit + fuzz tests. Host only; NOT road-tested."""
from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib import longcontrol as LCM
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc as LM
from openpilot.sunnypilot.selfdrive.controls.lib import lite_e2e_stop as ES
from openpilot.sunnypilot.selfdrive.controls.lib import lite_fusion as LF
from openpilot.sunnypilot.selfdrive.controls.lib import lite_launch as LL
from openpilot.sunnypilot.selfdrive.controls.lib import lite_launch_boost as LB

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3]
P = log.LongitudinalPersonality
LongCtrlState = LCM.LongCtrlState


def lead(d=9.3, v=0.6, a=0.3, present=True, radar=True):
  return SimpleNamespace(present=present, dRel=d, vLead=v, aLeadK=a, radar=radar)


def mlead(prob=0.9, x=10.8, v=0.6):
  return SimpleNamespace(prob=prob, x=[x], v=[v], a=[0.0])


def make_lc():
  CP = SimpleNamespace(stopAccel=-2.0, vEgoStopping=0.25, vEgoStarting=0.25,
                       longitudinalTuning=SimpleNamespace(kiBP=[0.], kiV=[0.]))
  return LCM.LongControl(CP, SimpleNamespace(enableGasInterceptor=False))


def cs(v, standstill=False, a=0.0):
  return SimpleNamespace(vEgo=v, aEgo=a, brakePressed=False, standstill=standstill,
                         cruiseState=SimpleNamespace(standstill=False))


class TestFusion(unittest.TestCase):
  def test_radar_alone_when_no_vision(self):
    self.assertTrue(LF.lead_moving_agreed(lead(), None))
    self.assertTrue(LF.lead_moving_agreed(lead(), mlead(prob=0.2, v=0.0)))

  def test_vision_only_never_agrees(self):
    self.assertFalse(LF.lead_moving_agreed(lead(radar=False), None))
    self.assertFalse(LF.lead_moving_agreed(lead(radar=False), mlead(v=3.0)))
    self.assertFalse(LF.lead_moving_agreed(SimpleNamespace(present=True, dRel=9, vLead=1, aLeadK=1), None))  # no flag

  def test_radar_only_launch_decision(self):
    # sunnypilot-lite-test2 (Matt): vision removed from the launch decision; radar lead moving = go
    self.assertFalse(LF.REQUIRE_MODEL_AGREE)
    self.assertTrue(LF.REQUIRE_RADAR)
    self.assertTrue(LF.lead_moving_agreed(lead(), mlead(v=0.0)))  # the model disagreeing no longer blocks
    self.assertTrue(LF.lead_moving_agreed(lead(), mlead(v=0.3)))
    self.assertFalse(LF.lead_moving_agreed(lead(present=False), mlead(v=0.3)))

  def test_model_stop_blocks(self):
    self.assertFalse(LF.model_stop_blocks(None, 9.0, 3.0))       # no planned stop
    self.assertFalse(LF.model_stop_blocks(6.0, 9.0, 3.0))        # stop behind the lead: explained
    self.assertFalse(LF.model_stop_blocks(2.0, 12.0, 0.5))       # lead has not moved far yet
    self.assertTrue(LF.model_stop_blocks(2.0, 12.0, 1.5))        # line the lead left behind
    self.assertTrue(LF.model_stop_blocks(float('nan'), 12.0, 1.5))  # bad data: block (conservative)

  def test_conservative_gap(self):
    self.assertFalse(LF.USE_MODEL_GAP)  # shipped off (sim regression)
    self.assertAlmostEqual(LF.conservative_gap(lead(d=10.0), mlead(x=9.0)), 10.0)
    LF.USE_MODEL_GAP = True
    try:
      self._conservative_gap_on()
    finally:
      LF.USE_MODEL_GAP = False

  def _conservative_gap_on(self):
    self.assertAlmostEqual(LF.conservative_gap(lead(d=10.0), None), 10.0)
    self.assertAlmostEqual(LF.conservative_gap(lead(d=10.0), mlead(x=9.0)), 9.0 - LF.RADAR_TO_CAMERA)
    self.assertAlmostEqual(LF.conservative_gap(lead(d=10.0), mlead(x=20.0)), 10.0)
    self.assertAlmostEqual(LF.conservative_gap(lead(d=10.0), mlead(prob=0.1, x=1.0)), 10.0)

  def test_fuzz_no_exceptions(self):
    r = np.random.default_rng(0)
    for _ in range(3000):
      vals = [r.normal(0, 20), float('nan'), float('inf'), -float('inf')]
      ld = lead(d=r.choice(vals), v=r.choice(vals), a=r.choice(vals), present=bool(r.random() < 0.8), radar=bool(r.random() < 0.7))
      ml = None if r.random() < 0.2 else SimpleNamespace(prob=r.choice(vals), x=[r.choice(vals)] if r.random() < 0.9 else [],
                                                          v=[r.choice(vals)] if r.random() < 0.9 else [], a=[])
      ok = LF.lead_moving_agreed(ld, ml)
      self.assertIsInstance(ok, bool)
      if ok and ld.present:
        for flag in (False, True):
          LF.USE_MODEL_GAP = flag
          g = LF.conservative_gap(ld, ml)
          if math.isfinite(ld.dRel):
            self.assertLessEqual(g, ld.dRel)
        LF.USE_MODEL_GAP = False


class TestLaunchFusionGate(unittest.TestCase):
  def setUp(self):
    LL.ENABLED = True
    self.l = LL.LiteLaunch()
    for _ in range(5):
      self.l.update(lead(d=9.05, v=0.0, a=0.0), 0.0, True, False, 0.0)

  def _go(self, ld, ml):
    out = None
    for _ in range(3):
      out = self.l.update(ld, 0.0, True, False, 0.05, None, False, model_lead=ml)
    return out

  def test_radar_alone_launches(self):
    self.assertTrue(self._go(lead(d=9.5, v=0.6, a=0.5), None)[0])

  def test_vision_disagreeing_no_longer_blocks(self):
    self.assertTrue(self._go(lead(d=9.5, v=0.6, a=0.5), mlead(v=0.0))[0])

  def test_vision_only_blocks(self):
    self.assertFalse(self._go(lead(d=9.5, v=0.6, a=0.5, radar=False), mlead(v=0.6))[0])

  def test_vision_closer_blocks_when_model_gap_on(self):
    LF.USE_MODEL_GAP = True
    try:
      # radar says 9.5 m but the model sees the lead at 7 m: conservative gap is under the stop target
      self.assertFalse(self._go(lead(d=9.5, v=0.6, a=0.5), mlead(x=7.0 + LF.RADAR_TO_CAMERA, v=0.6))[0])
    finally:
      LF.USE_MODEL_GAP = False

  def test_planner_passes_model_lead(self):
    src = (PKG / "selfdrive/controls/lib/longitudinal_planner.py").read_text()
    self.assertIn("model_lead = model_leads[0]", src)
    self.assertIn("output_should_stop_e2e, model_lead=model_lead,", src)
    self.assertIn("model_stop_x=model_stop_x, a_plan_radar=a_plan_radar", src)


class TestSmoothStopLongcontrol(unittest.TestCase):
  def _stop(self, smooth, a_target=-0.2, v0=0.29):
    LCM.STOP_SMOOTH = smooth
    try:
      lc = make_lc()
      lc.long_control_state = LongCtrlState.pid
      lc.last_output_accel = a_target
      lc.v_peak = 5.0  # came from a real approach
      v = v0; outs = []; vs = []
      for _ in range(400):
        out = lc.update(True, cs(v, standstill=v < 1e-3, a=-0.3), a_target, True, (-3.5, 2.0))
        outs.append(out); vs.append(v)
        v = max(0.0, v + out * 0.01)
      return np.array(outs), np.array(vs)
    finally:
      LCM.STOP_SMOOTH = True

  def test_gentle_at_stop_and_full_hold_after(self):
    o_new, v_new = self._stop(True)
    o_old, v_old = self._stop(False)
    i_new = int(np.argmax(v_new <= 0.0)); i_old = int(np.argmax(v_old <= 0.0))
    self.assertGreater(o_new[i_new - 1], -0.35)       # ~-0.15..-0.3 at the moment of stopping
    self.assertLess(o_old[i_old - 1], -0.5)           # stock grabs
    self.assertAlmostEqual(o_new[-1], -2.0, places=3)  # full standstill hold still builds
    d_new = float(np.sum(v_new) * 0.01); d_old = float(np.sum(v_old) * 0.01)
    self.assertLess(d_new - d_old, LM.STOP_TAPER_COMP)  # extra roll covered by the planned compensation

  def test_urgent_braking_passes_through(self):
    LCM.STOP_SMOOTH = True
    lc = make_lc(); lc.long_control_state = LongCtrlState.pid; lc.last_output_accel = -0.2; lc.v_peak = 5.0
    lc.update(True, cs(0.25, a=-0.3), -0.3, True, (-3.5, 2.0))
    self.assertTrue(lc.stop_soft)
    out = lc.update(True, cs(0.25, a=-0.3), -2.5, True, (-3.5, 2.0))
    self.assertLessEqual(out, -2.5 + 1e-6)

  def test_creep_keeps_stock_ramp(self):
    lc = make_lc(); lc.long_control_state = LongCtrlState.pid; lc.last_output_accel = 0.1
    lc.update(True, cs(0.28, a=0.05), 0.05, True, (-3.5, 2.0))
    self.assertFalse(lc.stop_soft)
    out = lc.update(True, cs(0.28, a=0.05), 0.05, True, (-3.5, 2.0))
    self.assertAlmostEqual(out, -2 * 1.0 * LCM.DT_CTRL, places=6)  # stock ramp from 0

  def test_short_creep_keeps_stock_ramp(self):
    lc = make_lc(); lc.long_control_state = LongCtrlState.pid; lc.last_output_accel = -0.2
    lc.update(True, cs(0.0, standstill=True), 0.0, False, (-3.5, 2.0))
    lc.long_control_state = LongCtrlState.pid
    lc.update(True, cs(0.6, a=-0.3), -0.3, False, (-3.5, 2.0))
    lc.update(True, cs(0.25, a=-0.3), -0.3, True, (-3.5, 2.0))
    self.assertFalse(lc.stop_soft)

  def test_never_positive_while_stopping(self):
    r = np.random.default_rng(1)
    for _ in range(2000):
      lc = make_lc(); lc.long_control_state = LongCtrlState.stopping; lc.stop_soft = bool(r.random() < 0.5)
      lc.last_output_accel = float(r.uniform(-3.5, 2.0))
      out = lc.update(True, cs(float(r.uniform(0, 0.3)), standstill=bool(r.random() < 0.3)), float(r.uniform(-3.5, 2.0)),
                      True, (-3.5, 2.0))
      self.assertLessEqual(out, 0.0)
      self.assertTrue(math.isfinite(out))

  def test_lead_stop_extra(self):
    self.assertAlmostEqual(float(LM.get_lead_stop_extra(0.0)), 10 * 0.3048, places=5)
    self.assertAlmostEqual(float(LM.get_lead_stop_extra(0.0, LM.STOP_TAPER_COMP)), 10 * 0.3048 + LM.STOP_TAPER_COMP, places=5)
    self.assertAlmostEqual(float(LM.get_lead_stop_extra(5.0, LM.STOP_TAPER_COMP)), 0.0, places=6)
    old = LM.LEAD_STOP_EXTRA
    try:
      LM.LEAD_STOP_EXTRA = 0.0
      self.assertEqual(float(LM.get_lead_stop_extra(0.0, 0.15)), 0.0)  # stock stays stock
    finally:
      LM.LEAD_STOP_EXTRA = old

  def test_stop_comp_latch(self):
    m = SimpleNamespace()
    f = LM.LongitudinalMpc.update_stop_comp
    seq = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.0),            # creep: never
           (3.0, LM.STOP_TAPER_COMP), (0.5, LM.STOP_TAPER_COMP),       # real approach
           (0.0, LM.STOP_TAPER_COMP), (0.0, LM.STOP_TAPER_COMP),       # held while stopped (no inching)
           (0.3, 0.0), (0.0, 0.0)]                                    # pulled away: creep target is base
    for v, want in seq:
      self.assertEqual(f(m, v), want, (v, want))


class TestSmoothLineStop(unittest.TestCase):
  def test_taper_ends_gentle(self):
    self.assertAlmostEqual(ES.stop_accel(0.0, ES.taper_dist(0.0) + 1e-3), 0.0, places=3)
    # following the taper law from 2.5 m/s with exactly the needed room: ends near -TAPER_C
    v, x, a_last = ES.TAPER_V, ES.taper_dist(ES.TAPER_V), None
    for _ in range(4000):
      a = ES.stop_accel(v, x)
      if v <= 0.05:
        break
      a_last = a; x -= v * 0.005; v = max(0.0, v + a * 0.005)
    self.assertGreater(a_last, -0.4)
    self.assertGreater(x, -0.05)  # did not pass the target

  def test_monotone_in_distance_and_never_positive(self):
    r = np.random.default_rng(2)
    for _ in range(3000):
      v = float(r.uniform(0, 20)); d1 = float(r.uniform(0.05, 80)); d2 = d1 + float(r.uniform(0, 20))
      a1, a2 = ES.stop_accel(v, d1), ES.stop_accel(v, d2)
      self.assertLessEqual(a1, 0.0); self.assertGreaterEqual(a1, -ES.MAX_EXTRA_DECEL - 1e-9)
      self.assertLessEqual(a1, a2 + 1e-9)  # less room never means less braking
      self.assertTrue(math.isfinite(a1))

  def test_stops_within_target_closed_loop(self):
    for v0 in (5.0, 8.0, 11.0, 13.0):
      v, x = v0, 60.0
      tgt = 60.0 - 0.0
      pos = 0.0
      while v > 0.05:
        a = ES.stop_accel(v, max(tgt - pos, 0.05))
        pos += v * 0.01; v = max(0.0, v + a * 0.01)
      self.assertLessEqual(pos, tgt + 0.10, v0)  # 10 ms Euler


class TestE2eStopMemory(unittest.TestCase):
  def _model(self, x_stop, n=33):
    t = np.linspace(0, 10, n)
    v = np.where(np.linspace(0, 60, n) < x_stop, 1.0, 0.0)
    x = np.minimum(t * 1.0, x_stop)
    return SimpleNamespace(position=SimpleNamespace(x=list(x), t=list(t)), velocity=SimpleNamespace(x=list(np.where(x < x_stop, 1.0, 0.0))))

  def test_target_does_not_collapse_but_never_beyond_model_stop(self):
    st = ES.E2eStopState()
    ES.shift_e2e_stop(self._model(8.0), None, 1.0, 0.0, False, state=st, dt=0.05)
    t1 = st.prev_target
    self.assertAlmostEqual(t1, ES.planned_stop_x(self._model(8.0)) - ES.EXTRA, places=6)
    # model replans its stop 1 m closer: target only moves by ego travel
    ES.shift_e2e_stop(self._model(7.0), None, 1.0, 0.0, False, state=st, dt=0.05)
    self.assertAlmostEqual(st.prev_target, t1 - 0.05, places=6)
    # never past the model's own current stop point
    ES.shift_e2e_stop(self._model(1.0), None, 1.0, 0.0, False, state=st, dt=0.05)
    self.assertLessEqual(st.prev_target, ES.planned_stop_x(self._model(1.0)) + 1e-9)
    # stop gone / lead explains -> forget
    ES.shift_e2e_stop(SimpleNamespace(position=SimpleNamespace(x=[0, 1], t=[0, 1]), velocity=SimpleNamespace(x=[5, 5])),
                      None, 1.0, 0.0, False, state=st)
    self.assertIsNone(st.prev_target)

  def test_never_raises_accel(self):
    r = np.random.default_rng(5)
    st = ES.E2eStopState()
    for _ in range(2000):
      if r.random() < 0.05:
        st.reset()
      a_in = float(r.uniform(-3, 2))
      a, _ = ES.shift_e2e_stop(self._model(float(r.uniform(0.5, 60))), None, float(r.uniform(0, 20)), a_in, bool(r.random() < 0.3),
                               state=st, dt=0.05)
      self.assertLessEqual(a, a_in + 1e-9)
      self.assertTrue(math.isfinite(a))


class TestBoost(unittest.TestCase):
  def test_shipped_on(self):
    self.assertTrue(LB.ENABLED)  # sunnypilot-lite-test2: re-enabled for a stronger 0-10 mph launch

  def test_disabled_is_passthrough(self):
    old = LB.ENABLED
    LB.ENABLED = False
    self.addCleanup(setattr, LB, "ENABLED", old)
    b = LB.LiteLaunchBoost()
    b.update(lead(d=9.05, v=0.0, a=0.0), 0.0, True, False, P.standard, 0.0, 0.0, 0.05)
    self.assertEqual(b.update(lead(d=14.0, v=3.0, a=1.5), 1.0, False, False, P.standard, 0.5, 0.5, 0.05), (False, 0.5))

  def test_speed_weight(self):
    self.assertEqual(LB.speed_weight(0.0), 1.0)
    self.assertEqual(LB.speed_weight(LB.FULL_V), 1.0)
    self.assertEqual(LB.speed_weight(LB.TAPER_END_V), 0.0)
    vs = np.linspace(0, 15, 300)
    w = [LB.speed_weight(v) for v in vs]
    self.assertTrue(all(b <= a + 1e-12 for a, b in zip(w, w[1:])))
    self.assertEqual(LB.speed_weight(float('nan')), 0.0)

  def test_caps(self):
    self.assertLessEqual(max(LB.FULL.values()), 2.0)
    self.assertGreater(LB.FULL[P.aggressive], LB.FULL[P.standard])
    self.assertGreater(LB.FULL[P.standard], LB.FULL[P.relaxed])

  def test_gates_when_enabled(self):
    LB.ENABLED = True
    try:
      b = LB.LiteLaunchBoost()
      b.update(lead(d=9.05, v=0.0, a=0.0), 0.0, True, False, P.standard, 0.0, 0.0, 0.05)
      on, a = b.update(lead(d=14.0, v=3.0, a=1.5), 1.0, False, False, P.standard, 0.5, 0.5, 0.05)
      self.assertTrue(on); self.assertGreater(a, 0.5); self.assertLessEqual(a, 0.5 + LB.JERK_UP * 0.05 + 1e-9)
      # slowing lead disarms for the rest of the launch
      on, a = b.update(lead(d=14.0, v=3.0, a=-0.5), 1.0, False, False, P.standard, 0.5, 0.5, 0.05)
      self.assertFalse(on); self.assertEqual(a, 0.5)
      on, _ = b.update(lead(d=14.0, v=3.0, a=1.5), 1.0, False, False, P.standard, 0.5, 0.5, 0.05)
      self.assertFalse(on)
      # vision-only lead / fcw / a model stop the lead does not explain (line): nothing
      for kw in (dict(ld=lead(d=14.0, v=3.0, a=1.5, radar=False), fcw=False, sx=None),
                 dict(ld=lead(d=14.0, v=3.0, a=1.5), fcw=True, sx=None),
                 dict(ld=lead(d=14.0, v=3.0, a=1.5), fcw=False, sx=2.0)):
        b = LB.LiteLaunchBoost()
        b.update(lead(d=9.05, v=0.0, a=0.0), 0.0, True, False, P.standard, 0.0, 0.0, 0.05)
        on, a = b.update(kw["ld"], 1.0, False, kw["fcw"], P.standard, 0.5, 0.5, 0.05, 2.0, False, model_stop_x=kw["sx"])
        self.assertFalse(on); self.assertEqual(a, 0.5)
      # radar-only: the model's own lead saying "stopped" or its shouldStop no longer block
      b = LB.LiteLaunchBoost()
      b.update(lead(d=9.05, v=0.0, a=0.0), 0.0, True, False, P.standard, 0.0, 0.0, 0.05)
      on, _ = b.update(lead(d=14.0, v=3.0, a=1.5), 1.0, False, False, P.standard, 0.5, 0.5, 0.05, 0.0, True,
                       model_lead=mlead(x=15.5, v=0.0))
      self.assertTrue(on)
    finally:
      LB.ENABLED = True

  def test_fuzz_never_lowers_never_exceeds(self):
    LB.ENABLED = True
    try:
      r = np.random.default_rng(3)
      b = LB.LiteLaunchBoost()
      for _ in range(5000):
        if r.random() < 0.1:
          b.reset()
        a_t = float(r.uniform(-3.5, 2.0)); a_prev = float(r.uniform(-3.5, 2.0))
        ld = lead(d=float(r.uniform(0, 60)), v=float(r.uniform(0, 20)), a=float(r.uniform(-4, 3)),
                  present=bool(r.random() < 0.9), radar=bool(r.random() < 0.8))
        ml = None if r.random() < 0.5 else mlead(prob=float(r.random()), x=float(r.uniform(0, 60)), v=float(r.uniform(0, 20)))
        on, a = b.update(ld, float(r.uniform(0, 15)), bool(r.random() < 0.2), bool(r.random() < 0.1), P.aggressive,
                         a_t, a_prev, 0.05, float(r.uniform(-3, 2.5)), bool(r.random() < 0.1), model_lead=ml)
        self.assertGreaterEqual(a, a_t)
        self.assertLessEqual(a, max(a_t, 2.0))
        self.assertTrue(math.isfinite(a))
    finally:
      LB.ENABLED = True


if __name__ == "__main__":
  unittest.main()
