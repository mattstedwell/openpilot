#!/usr/bin/env python3
"""t314u accel-curve host sim: cap table every 5 mph and no-lead speed-ups
(0-45, 25-45, 45-75 mph), before (e9e85baf79 C3X piecewise caps, Agg
+0.4 launch boost armed from a stop) vs after (this tree).

usage: test3_accel_curve_sim.py [OUT_JSON]
Cruise path only (free road): get_cruise_accel with this tree's J_CRUISE,
clipped to the cap, plus (after) the planner's positive-accel rise
limit. Plant: first-order tau 0.25 s at 100 Hz. Not road-tested.
"""
import json
import sys

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.tests.host_stubs import install as _install
_install()

import openpilot.selfdrive.controls.lib.longitudinal_planner as LP  # noqa: E402

MPH = 0.44704
PERS = ("aggressive", "standard", "relaxed")


def cap_before(v, p, boost_armed):
  stock = float(np.interp(v, [0., 10., 25., 40.], [1.6, 1.2, 0.8, 0.6]))
  if p == "relaxed":
    return 0.8 * stock
  extra = 0.4 * (1. - min(max(v, 0.), 10.) / 10.) if (p == "aggressive" and boost_armed) else 0.0
  return min(stock + extra, 2.0)


def cap_t312(v, p):
  """7b3f79ee93 (t312u): e9e85b caps, Agg/Std linear blend to Relaxed 40-45 mph."""
  own, rel = cap_before(v, p, False), cap_before(v, "relaxed", False)
  if p == "relaxed":
    return rel
  w = float(np.interp(v, [40 * MPH, 45 * MPH], [0., 1.]))
  return (1 - w) * own + w * rel


def cap_after(v, p):
  return float(LP.get_max_accel(v, p == "aggressive", p == "relaxed"))


def table():
  rows = []
  for m in range(0, 91, 5):
    v = m * MPH
    rows.append({"mph": m, **{f"{p}_before": round(cap_before(v, p, True), 3) for p in PERS},
                 **{f"{p}_after": round(cap_after(v, p), 3) for p in PERS}})
  return rows


def drive(p, v0_mph, v1_mph, after, dt=0.05, cap_fn=None, trace=None):
  v, a, a_cruise, a_out = v0_mph * MPH, 0.0, 0.0, 0.0
  v_set = v1_mph * MPH
  boost = v0_mph < 1.0  # e9e85b: Agg boost only arms at a stop
  t, peak_a, peak_j, peak_cmd_j, prev_a = 0.0, 0.0, 0.0, 0.0, 0.0
  while v < v_set - 0.05 and t < 120.0:
    if cap_fn is not None:
      cap = cap_fn(v, p)
    else:
      cap = cap_after(v, p) if after else cap_before(v, p, boost and v < 10.0)
    a_cruise = LP.get_cruise_accel(v_set, v, a_cruise, dt, cap=cap)
    a_new = float(np.clip(a_cruise, -3.5, cap))
    if after:
      a_new = float(LP.limit_accel_rise(a_new, a_out, v, dt))
    peak_cmd_j = max(peak_cmd_j, (a_new - a_out) / dt)
    a_out = a_new
    for _ in range(5):
      a += (a_out - a) * (1 - np.exp(-0.01 / 0.25))
      v += a * 0.01
      peak_j = max(peak_j, abs(a - prev_a) / 0.01)
      prev_a = a
    peak_a = max(peak_a, a)
    if trace is not None:
      trace.append((t, v / MPH, cap, a_out, a, (a_out - trace[-1][3]) / dt if trace else 0.0))
    t += dt
  return {"time_s": round(t, 2), "peak_accel": round(peak_a, 3), "peak_jerk": round(peak_j, 3),
          "peak_cmd_jerk": round(peak_cmd_j, 3)}


def main(out=None):
  res = {"caps": table(), "drives": {}}
  for p in PERS:
    res["drives"][p] = {}
    for a0, a1 in ((0, 45), (25, 45), (45, 75)):
      res["drives"][p][f"{a0}-{a1}"] = {"before": drive(p, a0, a1, False), "after": drive(p, a0, a1, True)}
  # t316u: accelerate 30 -> 55 mph through the highway blend; cap, accel
  # and jerk sampled every 2.5 mph. 'mid' = 7b3f79ee93 linear 40-45 blend.
  res["through_blend"] = {}
  for p in PERS:
    out_p = {}
    for label, fn, aft in (("mid_t312u", cap_t312, False), ("after", None, True)):
      tr = []
      d = drive(p, 30, 55, aft, cap_fn=fn, trace=tr)
      samples, nxt = [], 30.0
      for row in tr:
        if row[1] >= nxt:
          samples.append({"mph": round(nxt, 1), "cap": round(row[2], 3), "accel_cmd": round(row[3], 3),
                          "accel": round(row[4], 3)})
          nxt += 2.5
      caps = [r[2] for r in tr]
      cap_rate = max(abs(b - a) / 0.05 for a, b in zip(caps, caps[1:], strict=False))
      win = [(a, b) for a, b in zip(tr, tr[1:], strict=False) if 33.0 <= b[1] <= 52.0]
      win_jerk = max(abs(b[4] - a[4]) / 0.05 for a, b in win)
      win_cmd = max(abs(b[3] - a[3]) / 0.05 for a, b in win)
      out_p[label] = {"summary": {**d, "peak_cap_rate": round(cap_rate, 3), "jerk_33_52mph": round(win_jerk, 3),
                                  "cmd_jerk_33_52mph": round(win_cmd, 3)}, "samples": samples,
                      "cmd_jerk_max_abs": round(max(abs(r[5]) for r in tr[1:]), 3)}
    res["through_blend"][p] = out_p
  if out:
    with open(out, "w") as f:
      json.dump(res, f, indent=1)
  return res


if __name__ == "__main__":
  r = main(sys.argv[1] if len(sys.argv) > 1 else None)
  print("mph | Agg before/after | Std before/after | Rel before/after")
  for row in r["caps"]:
    print(f"{row['mph']:3d} | " + " | ".join(f"{row[p + '_before']:.2f}/{row[p + '_after']:.2f}" for p in PERS))
  for p, d in r["drives"].items():
    for k, ba in d.items():
      b, a = ba["before"], ba["after"]
      print(f"{p:10s} {k:6s} time {b['time_s']:5.1f}->{a['time_s']:5.1f} s  peak accel {b['peak_accel']:.2f}->{a['peak_accel']:.2f}  "
            f"peak jerk {b['peak_jerk']:.2f}->{a['peak_jerk']:.2f}  cmd jerk {b['peak_cmd_jerk']:.2f}->{a['peak_cmd_jerk']:.2f}")
  print("\n30->55 mph through the highway blend (mid = 7b3f79 linear 40-45, after = this tree)")
  for p, d in r["through_blend"].items():
    m, a = d["mid_t312u"], d["after"]
    print(f"{p}: peak cap rate {m['summary']['peak_cap_rate']:.3f}->{a['summary']['peak_cap_rate']:.3f} m/s³, "
          f"jerk in 33-52 mph {m['summary']['jerk_33_52mph']:.3f}->{a['summary']['jerk_33_52mph']:.3f} "
          f"(cmd {m['summary']['cmd_jerk_33_52mph']:.3f}->{a['summary']['cmd_jerk_33_52mph']:.3f}), "
          f"peak jerk incl. start {m['summary']['peak_jerk']:.2f}->{a['summary']['peak_jerk']:.2f}, "
          f"max |cmd jerk| {m['cmd_jerk_max_abs']:.2f}->{a['cmd_jerk_max_abs']:.2f}, time {m['summary']['time_s']}->{a['summary']['time_s']} s")
    for sm, sa in zip(m["samples"], a["samples"], strict=False):
      print(f"   {sm['mph']:5.1f} mph cap {sm['cap']:.3f}->{sa['cap']:.3f} accel {sm['accel']:.3f}->{sa['accel']:.3f}")
