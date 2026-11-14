"""Host (x86) acados long_mpc build for dev-c4-sunnypilot-lite sims.

The tree's prebuilt acados solver is aarch64-only. This generates the OCP
from this tree's long_mpc.py into BUILD, compiles it with gcc, and installs
a stand-in `acados_ocp_solver_pyx` module so long_mpc imports normally.
Host sims only; nothing here runs on a device. Adapted from devc4-test3
test3_c3x_equiv_sim.py _setup().
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


def setup(build: Path, root: Path):
  build = Path(build); root = Path(root)
  build.mkdir(parents=True, exist_ok=True)
  for p in (str(root / "opendbc_repo"), str(root / "msgq_repo"), str(root / "openpilot"), str(root)):
    if p not in sys.path:
      sys.path.insert(0, p)
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

  mpc_py = root / "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py"
  code = mpc_py.read_text()
  code = code.replace("from openpilot.third_party.acados.acados_template import", "from acados.acados_template import")
  code = code.replace("if __name__ == '__main__':  # generating code", "if True:")
  code = code.replace('EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")', f'EXPORT_DIR = "{build}/c_generated_code"')
  code = code.replace('JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")', f'JSON_FILE = "{build}/acados_ocp_long.json"')
  code = code.split('if __name__ == "__main__":')[0]
  g = {"__name__": "sp_lite_host_long_mpc", "__file__": str(build / "long_mpc.py")}
  exec(compile(code, "sp_lite_host_long_mpc", "exec"), g)
  ocp = g["gen_long_ocp"]()
  cwd = os.getcwd()
  os.chdir(build)
  try:
    gen = build / "c_generated_code"
    so = gen / "libacados_ocp_solver_long.so"
    if not so.exists():
      g["AcadosOcpSolver"].generate(ocp, json_file=g["JSON_FILE"])
      srcs = [str(gen / "acados_solver_long.c")] + [
        str(p) for d in ("long_model", "long_cost", "long_constraints") for p in sorted((gen / d).glob("*.c"))]
      inc = inst / "include"
      subprocess.check_call(["gcc", "-O2", "-fPIC", "-shared", "-o", str(so), *srcs, f"-I{gen}", f"-I{inc}",
                             f"-I{inc}/acados", f"-I{inc}/blasfeo/include", f"-I{inc}/hpipm/include",
                             f"-L{inst}/lib", "-lacados", "-lhpipm", "-lblasfeo", "-lm", f"-Wl,-rpath,{inst}/lib"])
  finally:
    os.chdir(cwd)

  class AcadosOcpSolverCython:
    def __new__(cls, *a, **k):
      os.chdir(build)
      try:
        return g["AcadosOcpSolver"](ocp, json_file=g["JSON_FILE"], build=False, generate=False, verbose=False)
      finally:
        os.chdir(cwd)

  name = "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx"
  mod = types.ModuleType(name)
  mod.AcadosOcpSolverCython = AcadosOcpSolverCython
  sys.modules[name] = mod
  pkg = "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code"
  if pkg not in sys.modules:
    m = types.ModuleType(pkg); m.__path__ = []
    sys.modules[pkg] = m
  return g


def radar(d, v, a, tau=1.5, present=True):
  from openpilot.cereal import log
  rs = log.RadarState.new_message()
  ld = rs.leadOne
  ld.present = present
  if present:
    ld.dRel = float(d); ld.vLead = float(v); ld.vLeadK = float(v); ld.aLeadK = float(a)
    ld.aLeadTau = float(tau); ld.modelProb = 1.0; ld.vRel = 0.0
  rs.leadTwo.present = False
  return rs
