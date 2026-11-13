#!/usr/bin/env python3
"""Host (x86) Chill longitudinal equivalence sims.

usage: test3_c3x_equiv_sim.py REPO [BUILD_DIR]
Prints one JSON object. Run on this official-dev rebuild and compare to the old test3 / C3X JSONs.
Accel caps use this tree's stock (2.0); T_FOLLOW and stop extras match C3X. experimental_mode is always False (Chill). Not road-tested.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path
from unittest.mock import MagicMock

warnings.filterwarnings("ignore")

_arg_repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
BUILD = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/test3dev_mpc_build")
# Worktree root (has openpilot/selfdrive/...) or a layout with selfdrive/ at top.
if (_arg_repo / "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py").exists():
  ROOT = _arg_repo
  REPO = _arg_repo / "openpilot"  # package dir
elif (_arg_repo / "selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py").exists():
  REPO = _arg_repo
  ROOT = _arg_repo.parent if (_arg_repo / "selfdrive").exists() else _arg_repo
else:
  raise SystemExit(f"could not find long_mpc.py under {_arg_repo}")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))


def _install_host_stubs():
  import sys as _sys, types as _types
  if "openpilot.common.params_pyx" not in _sys.modules:
    pyx = _types.ModuleType("openpilot.common.params_pyx")

    class Params:
      def get_bool(self, key):
        return False

      def get(self, key, *args, **kwargs):
        return None

      def put(self, *args, **kwargs):
        pass

      def put_bool(self, *args, **kwargs):
        pass

    pyx.Params = Params
    pyx.ParamKeyFlag = object
    pyx.ParamKeyType = object
    pyx.UnknownKeyName = type("UnknownKeyName", (Exception,), {})
    _sys.modules["openpilot.common.params_pyx"] = pyx
  if "msgq.ipc_pyx" not in _sys.modules:
    ipc = _types.ModuleType("msgq.ipc_pyx")

    class _Dummy:
      def __init__(self, *a, **k):
        pass

      def __getattr__(self, name):
        return lambda *a, **k: None

    class _Exc(Exception):
      pass

    ipc.Context = _Dummy
    ipc.Poller = _Dummy
    ipc.SubSocket = _Dummy
    ipc.PubSocket = _Dummy
    ipc.SocketEventHandle = _Dummy
    ipc.MultiplePublishersError = _Exc
    ipc.IpcError = _Exc
    ipc.toggle_fake_events = lambda *a, **k: None
    ipc.set_fake_prefix = lambda *a, **k: None
    ipc.get_fake_prefix = lambda *a, **k: b""
    ipc.delete_fake_prefix = lambda *a, **k: None
    ipc.wait_for_one_event = lambda *a, **k: None
    _sys.modules["msgq.ipc_pyx"] = ipc


def _setup(build: Path, repo: Path):
  _install_host_stubs()
  _cdll = ctypes.CDLL

  def cdll(p, *a, **k):
    if "libparams_c" in str(p):
      return MagicMock()
    return _cdll(p, *a, **k)

  ctypes.CDLL = cdll

  import acados
  inst = Path(acados.__file__).parent / "install"
  for lib in ("libblasfeo.so", "libhpipm.so", "libqpOASES_e.so.3.1", "libacados.so"):
    if (inst / "lib" / lib).exists():
      _cdll(str(inst / "lib" / lib), mode=ctypes.RTLD_GLOBAL)
  src_dir = build / "acados_src"
  (src_dir / "interfaces" / "acados_template").mkdir(parents=True, exist_ok=True)
  for name in ("lib", "include", "bin"):
    if not (src_dir / name).exists():
      (src_dir / name).symlink_to(inst / name)
  tmpl = src_dir / "interfaces" / "acados_template" / "acados_template"
  if not tmpl.exists():
    tmpl.symlink_to(Path(acados.__file__).parent / "acados_template")
  os.environ["ACADOS_SOURCE_DIR"] = str(src_dir)

  # `repo` here is the worktree root or the package dir; try both.
  mpc_py = Path(repo) / "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py"
  if not mpc_py.exists():
    mpc_py = Path(repo) / "selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py"
  code = mpc_py.read_text()
  code = code.replace(
    "from openpilot.third_party.acados.acados_template import",
    "from acados.acados_template import",
  )
  code = code.replace("if __name__ == '__main__':  # generating code", "if True:")
  code = code.replace(
    'EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")',
    f'EXPORT_DIR = "{build}/c_generated_code"',
  )
  code = code.replace(
    'JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")',
    f'JSON_FILE = "{build}/acados_ocp_long.json"',
  )
  code = code.split('if __name__ == "__main__":')[0]
  g = {"__name__": "test3_host_long_mpc", "__file__": str(build / "long_mpc.py")}
  exec(compile(code, "test3_host_long_mpc", "exec"), g)
  ocp = g["gen_long_ocp"]()
  cwd = os.getcwd()
  os.chdir(build)
  try:
    gen = build / "c_generated_code"
    so = gen / "libacados_ocp_solver_long.so"
    if not so.exists():
      g["AcadosOcpSolver"].generate(ocp, json_file=g["JSON_FILE"])
      srcs = [str(gen / "acados_solver_long.c")] + [
        str(p)
        for d in ("long_model", "long_cost", "long_constraints")
        for p in sorted((gen / d).glob("*.c"))
      ]
      inc = inst / "include"
      subprocess.check_call([
        "gcc", "-O2", "-fPIC", "-shared", "-o", str(so), *srcs,
        f"-I{gen}", f"-I{inc}", f"-I{inc}/acados",
        f"-I{inc}/blasfeo/include", f"-I{inc}/hpipm/include",
        f"-L{inst}/lib", "-lacados", "-lhpipm", "-lblasfeo", "-lm",
        f"-Wl,-rpath,{inst}/lib",
      ])
  finally:
    os.chdir(cwd)

  class AcadosOcpSolverCython:
    def __new__(cls, *a, **k):
      os.chdir(build)
      try:
        return g["AcadosOcpSolver"](
          ocp, json_file=g["JSON_FILE"], build=False, generate=False, verbose=False
        )
      finally:
        os.chdir(cwd)

  # After exec, the tree's long_mpc functions live in g. Re-import through
  # the normal package so planner hooks resolve, with the host solver stubbed.
  name = "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx"
  mod = types.ModuleType(name)
  mod.AcadosOcpSolverCython = AcadosOcpSolverCython
  sys.modules[name] = mod
  pkg = "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code"
  if pkg not in sys.modules:
    m = types.ModuleType(pkg)
    m.__path__ = []
    sys.modules[pkg] = m
  return g


def _radar(d, v, a, tau=1.5, present=True):
  from openpilot.cereal import log
  rs = log.RadarState.new_message()
  ld = rs.leadOne
  ld.present = present
  if hasattr(ld, "status"):
    ld.status = present
  if present:
    ld.dRel = float(d)
    ld.vLead = float(v)
    ld.vLeadK = float(v)
    ld.aLeadK = float(a)
    ld.aLeadTau = float(tau)
    ld.modelProb = 1.0
    ld.vRel = 0.0
  rs.leadTwo.present = False
  if hasattr(rs.leadTwo, "status"):
    rs.leadTwo.status = False
  return rs


def _plan(mpc, ev, plan_a, v_des, rs, v_cruise, personality, exp_mode, cap, dt_mdl=0.05,
         a_cruise_prev=None):
  """MPC + cruise candidate, matching this tree's ACC planner.

  Official sunnypilot/dev removed the cruise obstacle from the MPC;
  set-speed tracking is get_cruise_accel, and the plan is min(lead-MPC,
  cruise). `cap` is the personality cruise ceiling (2.0 stock, 1.6 Relaxed).
  """
  import numpy as np
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
  from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
  from openpilot.selfdrive.modeld.constants import ModelConstants
  from openpilot.selfdrive.controls.lib.longitudinal_planner import get_cruise_accel, choose_acc_plan

  ctrl_t = ModelConstants.T_IDXS[:CONTROL_N]
  mpc.set_weights(not (ev < 0.05), personality=personality, experimental_mode=exp_mode)
  mpc.set_cur_state(v_des, plan_a)
  mpc.update(rs, personality=personality, experimental_mode=exp_mode)
  vt = np.interp(ctrl_t, T_IDXS_MPC, mpc.v_solution)
  at = np.interp(ctrl_t, T_IDXS_MPC, mpc.a_solution)
  a_mpc = get_accel_from_plan(vt, at, ctrl_t, action_t=0.05 + dt_mdl)
  ss_mpc = should_stop(ev, a_mpc)
  if a_cruise_prev is None:
    a_cruise_prev = plan_a
  a_cruise = get_cruise_accel(v_cruise, ev, a_cruise_prev, dt_mdl, cap=cap)
  ss_cruise = should_stop(ev, a_cruise)
  a_t, source, ss = choose_acc_plan(a_mpc, mpc.source, ss_mpc, a_cruise, ss_cruise)
  return float(a_t), bool(ss), float(a_cruise)


def main():
  import numpy as np
  from openpilot.cereal import log

  g = _setup(BUILD, ROOT)
  # Import after solver stub is in place.
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
    LongitudinalMpc, get_T_FOLLOW, get_jerk_factor, get_lead_stop_extra,
    get_safe_obstacle_distance, STOP_DISTANCE,
  )
  from opendbc.car.interfaces import ACCEL_MAX
  from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel
  from openpilot.sunnypilot.selfdrive.car.chill_aggressive import ChillAggressiveLaunch

  P = log.LongitudinalPersonality
  personalities = {
    "aggressive": P.aggressive,
    "standard": P.standard,
    "relaxed": P.relaxed,
  }
  MPH = 0.44704
  EXP = False  # Chill
  dt_mdl, dt_ctrl, tau = 0.05, 0.01, 0.25

  def is_relaxed(p):
    return p == P.relaxed

  def accel_cap(v, p, launch):
    return float(get_max_accel(v, launch, is_relaxed(p)))

  def time_to(p, mph0, mph1, launch=False, dt=0.001):
    v0, v1 = mph0 * MPH, mph1 * MPH
    v, t = v0, 0.0
    while v < v1 and t < 180.0:
      a = accel_cap(v, p, launch)
      if a <= 0.01:
        return None
      v += a * dt
      t += dt
    return t

  policy = {}
  for name, p in personalities.items():
    speeds = [0, 20, 30, 45, 65]
    policy[name] = {
      "jerk": get_jerk_factor(p, EXP),
      "stop_extra_m": get_lead_stop_extra(p, EXP),
      "t_follow": {str(mph): get_T_FOLLOW(p, EXP, mph * MPH) for mph in speeds},
      "accel_cap": {str(mph): accel_cap(mph * MPH, p, launch=(p == P.aggressive)) for mph in (0, 10, 20, 22, 30, 45, 60, 75)},
      "follow_gap_m": {
        str(mph): get_safe_obstacle_distance(mph * MPH, get_T_FOLLOW(p, EXP, mph * MPH))
        for mph in speeds
      },
      "t_0_30": time_to(p, 0, 30, launch=(p == P.aggressive)),
      "t_55_75": time_to(p, 55, 75, launch=(p == P.aggressive)),
    }

  def sim_loop(personality, ev0, ex_lead0, lv0, la_fn, v_cruise, sim_t, d_cutin=None, cutin_t=None):
    launch = ChillAggressiveLaunch()
    mpc = LongitudinalMpc()
    ex, ev, ea = 0.0, ev0, 0.0
    lx, lv = ex_lead0, lv0
    plan_a, v_des, last, state, ss = (0.0 if ev0 > 0.3 else -0.5), ev0, (-0.5 if ev0 < 0.3 else 0.0), ("pid" if ev0 > 0.3 else "stopping"), ev0 < 0.25
    a_cruise_st = plan_a
    t_req = t_move = t_decel = None
    peak_a, peak_decel, peak_j, min_gap = -1e9, 1e9, 0.0, 1e9
    final_gap = None
    prev_a = plan_a
    traces_a = []
    for k in range(int(sim_t / dt_ctrl)):
      t = k * dt_ctrl
      la = la_fn(t, lv)
      lv = max(lv + la * dt_ctrl, 0.0)
      lx += lv * dt_ctrl
      present = True
      d_rel = lx - ex
      if d_cutin is not None:
        present = t >= cutin_t
        if present and k == int(cutin_t / dt_ctrl):
          # Place lead at d_cutin ahead, same lane, at given lv
          lx = ex + d_cutin
          d_rel = d_cutin
      if k % 5 == 0:
        rs = _radar(d_rel, lv, la, present=present)
        rs.leadOne.vRel = float((lv - ev) if present else 0.0)
        launch_active = launch.update(
          True, EXP, personality, ev, v_cruise, False
        )
        cap = accel_cap(ev, personality, launch_active)
        a_t, ss, a_cruise_st = _plan(mpc, ev, plan_a, v_des, rs, v_cruise, personality, EXP, cap,
                                     a_cruise_prev=a_cruise_st)
        if a_t > 0:
          a_t = min(a_t, cap)
        jerk = abs(a_t - prev_a) / dt_mdl
        peak_j = max(peak_j, jerk)
        prev_a = a_t
        plan_a = float(np.clip(a_t, -3.5, 2.5))
        v_des = 0.5 * max(0.0, v_des + dt_mdl * plan_a) + 0.5 * ev
        if ev < 0.25:
          ss = plan_a < 0.1
        traces_a.append((t, plan_a, ev, d_rel if present else None))
        if t_req is None and plan_a > 0.05:
          t_req = t
        if t_decel is None and plan_a < -0.05:
          t_decel = t
      # simple plant
      if state == "stopping" and (not ss):
        state = "pid"
      elif state == "pid" and ss and ev < 0.25:
        state = "stopping"
      if state == "stopping":
        out = last
        if out > -2.0:
          # Official sunnypilot/dev longcontrol: 1.0 m/s²/s (was 0.3 on
          # release-mici / C3X Toyota stoppingDecelRate).
          out = min(out, 0.0) - 1.0 * dt_ctrl
        out = max(out, -2.0)
      else:
        # first-order toward plan
        out = float(ea + (plan_a - ea) * (1.0 - np.exp(-dt_ctrl / tau)))
      last = out
      ea = out
      ev = max(0.0, ev + ea * dt_ctrl)
      ex += ev * dt_ctrl
      if t_move is None and ev > 0.447:  # 1 mph
        t_move = t
      if present:
        gap = (lx - ex)
        min_gap = min(min_gap, gap)
        final_gap = gap
      peak_a = max(peak_a, ea)
      peak_decel = min(peak_decel, ea)
    return {
      "t_req": t_req,
      "t_move": t_move,
      "t_decel": t_decel,
      "peak_a": peak_a if peak_a > -1e8 else None,
      "peak_decel": peak_decel if peak_decel < 1e8 else None,
      "peak_j": peak_j,
      "min_gap": min_gap if min_gap < 1e8 else None,
      "final_gap": final_gap,
      "end_v": ev,
    }

  sims = {}
  # Launch: ego stopped, desired standstill gap, lead pulls away at 1.0 m/s2
  for name, p in personalities.items():
    extra = get_lead_stop_extra(p, EXP)
    d0 = STOP_DISTANCE + extra
    sims[f"launch_{name}"] = sim_loop(
      p, ev0=0.0, ex_lead0=d0, lv0=0.0,
      la_fn=lambda t, lv: 1.0, v_cruise=25.0, sim_t=10.0,
    )
    sims[f"launch_{name}"]["d0"] = d0
    # Stopped lead, should not creep (stock C3X may creep - record honestly)
    sims[f"stopped_{name}"] = sim_loop(
      p, ev0=0.0, ex_lead0=d0, lv0=0.0,
      la_fn=lambda t, lv: 0.0, v_cruise=25.0, sim_t=8.0,
    )
    # Approach a stopped lead from 10 m/s, start 80 m back
    sims[f"stop_from_10ms_{name}"] = sim_loop(
      p, ev0=10.0, ex_lead0=80.0, lv0=0.0,
      la_fn=lambda t, lv: 0.0, v_cruise=25.0, sim_t=20.0,
    )
    # Follow at constant speeds: start at desired gap, lead holds speed
    for mph in (20, 30, 45, 65):
      v = mph * MPH
      T = get_T_FOLLOW(p, EXP, v)
      d = get_safe_obstacle_distance(v, T)
      sims[f"follow_{mph}_{name}"] = sim_loop(
        p, ev0=v, ex_lead0=d, lv0=v,
        la_fn=lambda t, lv: 0.0, v_cruise=v + 5.0, sim_t=12.0,
      )
      sims[f"follow_{mph}_{name}"]["desired_gap"] = d
      sims[f"follow_{mph}_{name}"]["T"] = T

    # Highway cut-in at 65 mph
    v = 65 * MPH
    T = get_T_FOLLOW(p, EXP, v)
    d_ss = get_safe_obstacle_distance(v, T)
    for headway, vrel_mph in ((0.6, 0.0), (0.9, 0.0), (0.6, -3.0), (0.6, -6.0)):
      d_ci = headway * v
      lv = v + vrel_mph * MPH
      key = f"cutin_65_h{headway}_vrel{vrel_mph}_{name}"
      sims[key] = sim_loop(
        p, ev0=v, ex_lead0=d_ss, lv0=lv,
        la_fn=lambda t, lv: 0.0, v_cruise=v + 2.0, sim_t=8.0,
        d_cutin=d_ci, cutin_t=1.0,
      )
      sims[key]["d_cutin"] = d_ci
      sims[key]["desired_gap"] = d_ss

  out = {
    "repo": str(REPO),
    "policy": policy,
    "sims": {k: {kk: (None if vv is None else (float(vv) if isinstance(vv, (int, float, np.floating)) else vv))
                 for kk, vv in v.items()} for k, v in sims.items()},
    "STOP_DISTANCE": STOP_DISTANCE,
    "stock_cruise_cap": ACCEL_MAX,
    "note": "Standard stock is this tree's official ACCEL_MAX=2.0 (flat); Relaxed is 0.8 of that. C3X used [1.6,1.2,0.8,0.6]. No cruise obstacle; _plan uses get_cruise_accel.",
  }
  print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
  # acados prints "compiled without OpenMP" to stdout; keep JSON clean.
  import os as _os
  out_path = _os.environ.get("TEST3_EQUIV_OUT")
  if out_path:
    data = None
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    # Re-run main but capture: actually main() prints. Patch print at end instead.
  # If a trailing path ending in .json is given after BUILD, write there.
  args = [a for a in sys.argv[1:] if a.endswith(".json")]
  # Re-exec main into a buffer
  import io
  from contextlib import redirect_stdout
  buf = io.StringIO()
  with redirect_stdout(buf):
    main()
  text = buf.getvalue()
  # drop non-json prefix
  i = text.find("{")
  if i < 0:
    sys.stderr.write(text)
    sys.exit(2)
  text = text[i:]
  dest = args[-1] if args else None
  dest = dest or _os.environ.get("TEST3_EQUIV_OUT")
  if dest:
    Path(dest).write_text(text)
    sys.stderr.write(f"wrote {dest} ({len(text)} bytes)\n")
  else:
    sys.stdout.write(text)
