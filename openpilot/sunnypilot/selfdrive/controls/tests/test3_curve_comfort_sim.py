#!/usr/bin/env python3
"""Curve comfort host sim (t311u): accel allowed vs lateral accel, and a
closed-loop drive through typical curves with/without CurveComfort.

usage: test3_curve_comfort_sim.py [OUT_JSON]
No lead (cruise accel path, which is where positive accel comes from
with a free road). Model plan is synthesized from the road curvature at
ego's speed. Plant: first-order tau 0.25 s. Not road-tested.
"""
import json
import sys

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install
_install()

from openpilot.cereal import log  # noqa: E402
from openpilot.selfdrive.modeld.constants import ModelConstants  # noqa: E402
import openpilot.selfdrive.controls.lib.longitudinal_planner as LP  # noqa: E402
from openpilot.sunnypilot.selfdrive.car.chill_aggressive import ChillAggressiveLaunch  # noqa: E402
from openpilot.sunnypilot.selfdrive.controls.lib.devc4_test3_curve_comfort import (  # noqa: E402
  CurveComfort, long_accel_allowed)

MPH = 0.44704
P = log.LongitudinalPersonality
T_IDXS = np.array(ModelConstants.T_IDXS)


def cap_for(personality, v):
  relaxed = personality == P.relaxed
  agg = personality == P.aggressive
  if hasattr(LP, "smooth_accel_cap"):
    return float(LP.get_max_accel(v, agg, relaxed))
  return float(LP.get_max_accel(v, agg and v < 10.0, relaxed))


def road(R, s_curve=150.0, arc=None, clothoid=40.0):
  """Curvature vs distance: straight, clothoid in, arc, clothoid out, straight."""
  k = 1.0 / R
  arc = arc if arc is not None else min(np.pi / 2 * R, 400.0)
  a0, a1, a2, a3 = s_curve, s_curve + clothoid, s_curve + clothoid + arc, s_curve + 2 * clothoid + arc
  def kappa(s):
    return float(np.interp(s, [0, a0, a1, a2, a3, a3 + 1e6], [0, 0, k, k, 0, 0]))
  return kappa, a3


def drive(personality, R, v0, v_set, use_curve, t_end=40.0, dt=0.05):
  # Curve starts 2 s ahead at the initial speed: tests accelerating into
  # and through a curve that ego enters at about its design speed.
  kappa, s_end = road(R, s_curve=2.0 * v0)
  cc = CurveComfort()
  s, v, a, a_cruise = 0.0, v0, 0.0, 0.0
  out = dict(peak_long_at_lat1=0.0, peak_total=0.0, peak_lat=0.0, v_curve_mph=None, exit_jerk=0.0,
             peak_long_in_curve=0.0, peak_jerk=0.0)
  prev_a = 0.0
  a_out = 0.0
  tr = []
  for i in range(int(t_end / dt)):
    t = i * dt
    k_now = kappa(s)
    yaw = np.array([kappa(s + v * tt) * v for tt in T_IDXS])
    vplan = np.full_like(T_IDXS, v)
    cap = cap_for(personality, v)
    if use_curve:
      cap = cc.update(cap, v, k_now, yaw, vplan, T_IDXS, dt)
    a_cruise = LP.get_cruise_accel(v_set, v, a_cruise, dt, cap=cap)
    a_cmd = float(np.clip(a_cruise, -3.5, cap))
    if hasattr(LP, "limit_accel_rise"):  # t314u planner rise limit
      a_cmd = float(LP.limit_accel_rise(a_cmd, a_out, v, dt))
    a_out = a_cmd
    for _ in range(5):
      a += (a_cmd - a) * (1 - np.exp(-0.01 / 0.25))
      v = max(0.0, v + a * 0.01)
      s += v * 0.01
    lat = v * v * kappa(s)
    j = (a - prev_a) / dt
    prev_a = a
    out["peak_lat"] = max(out["peak_lat"], lat)
    out["peak_total"] = max(out["peak_total"], float(np.hypot(max(a, 0.0), lat)))
    if lat >= 1.0:
      out["peak_long_at_lat1"] = max(out["peak_long_at_lat1"], a)
    if kappa(s) > 0:
      out["peak_long_in_curve"] = max(out["peak_long_in_curve"], a)
      if out["v_curve_mph"] is None and kappa(s) >= 0.99 / R:
        out["v_curve_mph"] = round(v / MPH, 1)
    out["peak_jerk"] = max(out["peak_jerk"], abs(j))
    if s > s_end - 40:
      out["exit_jerk"] = max(out["exit_jerk"], abs(j))
    if i % 10 == 0:
      tr.append([round(t, 1), round(s, 1), round(v / MPH, 1), round(a, 3), round(lat, 3), round(cap, 3)])
    if s > s_end + 300:
      break
  out = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in out.items()}
  out["trace"] = tr
  return out


CASES = [
  # (name, radius m, entry speed, set speed): ego enters at the entry
  # speed, below set speed, so stock would accelerate through the curve.
  ("freeway 65 mph, R=1000 m (entry lat 0.84)", 1000.0, 65 * MPH, 75 * MPH),
  ("freeway 65 mph, R=600 m (entry lat 1.41)", 600.0, 65 * MPH, 75 * MPH),
  ("on-ramp 40 mph, R=200 m (entry lat 1.60)", 200.0, 40 * MPH, 55 * MPH),
  ("on-ramp 35 mph, R=150 m (entry lat 1.63)", 150.0, 35 * MPH, 55 * MPH),
  ("loop 25 mph, R=60 m (entry lat 2.08)", 60.0, 25 * MPH, 45 * MPH),
  ("city turn 15 mph, R=20 m (entry lat 2.25)", 20.0, 15 * MPH, 30 * MPH),
]


def table():
  rows = []
  for lat in [0.0, 0.3, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0]:
    rows.append((lat, round(long_accel_allowed(lat), 3)))
  return rows


def main(out=None):
  res = {"allowed_vs_lat": table(), "drives": {}}
  for pname, p in (("aggressive", P.aggressive), ("standard", P.standard), ("relaxed", P.relaxed)):
    res["drives"][pname] = {}
    for name, R, v0, vs in CASES:
      b = drive(p, R, v0, vs, False)
      a = drive(p, R, v0, vs, True)
      res["drives"][pname][name] = {"before": b, "after": a}
  if out:
    with open(out, "w") as f:
      json.dump(res, f, indent=1)
  return res


if __name__ == "__main__":
  r = main(sys.argv[1] if len(sys.argv) > 1 else None)
  print("lat accel -> long accel allowed:", r["allowed_vs_lat"])
  for p, d in r["drives"].items():
    for n, ba in d.items():
      b, a = ba["before"], ba["after"]
      print(f"{p:10s} {n:40s} long@lat>=1: {b['peak_long_at_lat1']:.2f}->{a['peak_long_at_lat1']:.2f}  "
            f"in-curve long: {b['peak_long_in_curve']:.2f}->{a['peak_long_in_curve']:.2f}  "
            f"total: {b['peak_total']:.2f}->{a['peak_total']:.2f}  lat: {b['peak_lat']:.2f}->{a['peak_lat']:.2f}  "
            f"v_curve: {b['v_curve_mph']}->{a['v_curve_mph']} mph  jerk: {b['peak_jerk']:.2f}->{a['peak_jerk']:.2f}")
