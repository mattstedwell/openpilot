"""Host stubs for aarch64-only msgq, params, and acados extensions.

Import and call install() before any long_mpc or planner import so the
ported C3X Chill tests can run on an x86 host.
"""
from __future__ import annotations

import sys
import types


def install() -> None:
  if "openpilot.common.params_pyx" not in sys.modules:
    pyx = types.ModuleType("openpilot.common.params_pyx")

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
    sys.modules["openpilot.common.params_pyx"] = pyx

  if "msgq.ipc_pyx" not in sys.modules:
    ipc = types.ModuleType("msgq.ipc_pyx")

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
    sys.modules["msgq.ipc_pyx"] = ipc

  name = "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx"
  if name not in sys.modules:
    import numpy as np

    mod = types.ModuleType(name)

    class AcadosOcpSolverCython:
      def __init__(self, *a, **k):
        pass

      def cost_set(self, *a, **k):
        pass

      def set(self, *a, **k):
        pass

      def constraints_set(self, *a, **k):
        pass

      def solve(self):
        return 0

      def get_stats(self, k):
        return [0.0]

      def get(self, i, what):
        if what == "u":
          return np.zeros(1)
        return np.zeros(3)

    mod.AcadosOcpSolverCython = AcadosOcpSolverCython
    sys.modules[name] = mod
    pkg = "openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code"
    if pkg not in sys.modules:
      m = types.ModuleType(pkg)
      m.__path__ = []
      sys.modules[pkg] = m
