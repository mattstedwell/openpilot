#!/usr/bin/env python3
"""Monte-Carlo closed-loop sweep for sunnypilot-lite (host only, NOT road-tested).

usage: sp_lite_mc.py TREE BUILD OUT_JSON CAND_JSON PERSONALITY SEED0 N
Reuses sp_lite_launch_sim.run() with randomized scenarios: lead launch accel
0.5-3, lead braking -1..-5, initial gap, ego 0-45 mph, radar dropouts, radar
noise, vision flicker / disagreement, cut-ins, plant lag 0.1-0.4 s, extra
plant delay and slower brake release. Same seeds -> same scenarios for any tree.
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE / "sp_lite_launch_sim.py"
TREE, BUILD, OUT, CAND, PERS, SEED0, NRUN = sys.argv[1:8]
src = SIM.read_text().split('if os.environ.get("SIM_NO_MAIN"):')[0]
sys.argv = [str(SIM), TREE, BUILD, OUT, CAND]
ns = {"__file__": str(SIM), "__name__": "sp_lite_sim_mc"}
exec(compile(src, str(SIM), "exec"), ns)
np = ns["np"]; S = ns["S"]; MPH = ns["MPH"]; P = ns["P"]
pers = {"aggressive": P.aggressive, "standard": P.standard, "relaxed": P.relaxed}[PERS]


def scenario(seed):
  r = np.random.default_rng(seed)
  kind = r.choice(["launch", "launch", "follow_brake", "rollstop", "approach", "stopped_hold"])
  ev0 = 0.0; lv0 = 0.0; gap0 = None; cut = None
  if kind == "launch":
    a_up = r.uniform(0.5, 3.0); vt = r.uniform(5, 45) * MPH; a_dn = -r.uniform(1.0, 5.0)
    segs = [("hold", r.uniform(1.0, 4.0)), ("acc", a_up, vt), ("hold", r.uniform(0.0, 3.0)), ("acc", a_dn, 0.0), ("hold", 5.0)]
    gap0 = ns["stop_target"]() + r.uniform(-1.0, 3.0)
  elif kind == "follow_brake":
    ev0 = lv0 = r.uniform(3, 45) * MPH
    segs = [("hold", r.uniform(1.0, 4.0)), ("acc", -r.uniform(1.0, 5.0), r.choice([0.0, lv0 * r.uniform(0.0, 0.7)])),
            ("hold", r.uniform(1.0, 3.0)), ("acc", r.uniform(0.5, 3.0), lv0), ("hold", 4.0)]
    gap0 = ns["eq_gap"](pers, lv0) * r.uniform(0.6, 1.4)
  elif kind == "rollstop":
    n = int(r.integers(4, 9)); D = r.uniform(0.3, 5.0)
    segs = [("hold", 2.0)] + sum([[("dist", r.uniform(0.5, 2.0), -r.uniform(1.0, 4.0), D), ("hold", r.uniform(0.3, 3.0))]
                                 for _ in range(n)], [])
  elif kind == "approach":
    ev0 = r.uniform(5, 45) * MPH; lv0 = 0.0; gap0 = max(ev0 * ev0 / (2 * 2.5) + r.uniform(5.0, 60.0), 10.0)
    segs = [("hold", r.uniform(3.0, 8.0)), ("acc", r.uniform(0.5, 3.0), r.uniform(5, 30) * MPH), ("hold", 4.0)]
  else:
    segs = [("hold", r.uniform(5.0, 15.0)), ("acc", r.uniform(0.5, 3.0), r.uniform(5, 25) * MPH), ("hold", 3.0)]
  lost = []
  if r.random() < 0.5:
    for _ in range(int(r.integers(1, 4))):
      t0 = r.uniform(0.5, 15.0); lost.append((t0, t0 + r.uniform(0.1, 1.5)))
  if r.random() < 0.2 and kind in ("launch", "follow_brake"):
    cut = dict(t=r.uniform(2.0, 8.0), gap=r.uniform(3.0, 15.0), v=max(0.0, r.uniform(0, 15) * MPH))
  vision = None
  vr = r.random()
  if vr < 0.15:
    t0 = r.uniform(0, 8); vision = ("flicker", t0, t0 + r.uniform(0.5, 4.0))
  elif vr < 0.25:
    t0 = r.uniform(0, 8); vision = ("stopped", t0, t0 + r.uniform(0.3, 2.0))
  elif vr < 0.45:
    vision = ("fallback", 0.0, 1e9)
  sc = S(f"mc_{seed}_{kind}", segs, ev0=ev0, lv0=lv0, gap0=gap0, cut=cut, noise=bool(r.random() < 0.3), lost=lost,
         kind="mc", set_gap_check=cut is None, vision=vision, t_tail=6.0)
  sc["plant_tau"] = r.uniform(0.1, 0.4)
  sc["plant_delay_n"] = int(r.integers(3, 16))       # 0.03-0.15 s
  sc["brake_release_tau"] = sc["plant_tau"] * r.uniform(1.0, 1.8)
  sc["gone_after"] = None
  return sc


res = {"tree": TREE, "candidate": json.loads(CAND), "personality": PERS, "runs": {}}
for seed in range(int(SEED0), int(SEED0) + int(NRUN)):
  sc = scenario(seed)
  try:
    rr = ns["run"](pers, sc)
    rr = {k: v for k, v in rr.items() if k not in ("stop_comfort",)} | {
      "worst_stop_roll_jerk": max([s["peak_jerk_roll"] for s in rr["stop_comfort"]], default=None),
      "worst_a_at_stop": min([s["a_at_stop"] for s in rr["stop_comfort"]], default=None)}
    rr["name"] = sc["name"]; rr["cut"] = sc["cut"] is not None
  except Exception as e:  # noqa: BLE001
    rr = {"name": sc["name"], "error": repr(e)}
  res["runs"][str(seed)] = rr
Path(OUT).write_text(json.dumps(res))
print("done", OUT, len(res["runs"]))
