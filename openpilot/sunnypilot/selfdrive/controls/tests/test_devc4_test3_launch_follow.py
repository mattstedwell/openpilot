"""Closed-loop regression: launch take-up must not bypass the MPC lead plan.

Stop behind a lead; the lead pulls away at 1.0 m/s² to 30 mph and holds.
Aggressive must converge toward the T=1.25 desired bumper gap and never
go below the planned standstill gap (24.69 ft). Relaxed launch must
release (the pre-fix path stayed active for the whole 40 s).

Requires the host MPC solver at /tmp/test3dev_mpc_build (same as the
equivalence sims). Skips if that build is missing. Nothing road-tested.
"""
from pathlib import Path

import pytest

BUILD = Path("/tmp/test3dev_mpc_build")

pytestmark = pytest.mark.skipif(
  not (BUILD / "c_generated_code" / "libacados_ocp_solver_long.so").exists()
  and not (BUILD / "acados_ocp_long.json").exists(),
  reason="host MPC solver build not present",
)

MPH = 0.44704
FT = 0.3048


@pytest.fixture(scope="module")
def kw():
  from openpilot.sunnypilot.selfdrive.controls.tests.test3_launchfix_sim import (
    prepare, worktree_root,
  )
  return prepare(worktree_root(), BUILD)


def test_aggressive_converges_to_t125_desired_and_holds_standstill_floor(kw):
  from openpilot.sunnypilot.selfdrive.controls.tests.test3_launchfix_sim import (
    run_launch_follow,
  )
  P = kw["P"]
  r = run_launch_follow(P.aggressive, kw, target_mph=30, sim_t=40.0)
  # Planned standstill is 24.69 ft; the pre-fix path went to 16.8 ft.
  assert r["floor_ft"] == pytest.approx(24.69, abs=0.02)
  assert r["below_floor"] is False
  assert r["min_gap_ft"] >= 24.6
  # T is still official / C3X 1.25, not the reverted 1.5.
  assert r["T_at_30"] == pytest.approx(1.25)
  desired_30 = 1.25 * 30 * MPH + 6.0
  assert desired_30 / FT == pytest.approx(74.7, abs=0.2)
  # After launch hands off, the gap must settle near the 1.25 desired
  # (pre-fix was 28.2 ft / 0.64 s at the 30 mph crossing, and Relaxed
  # stayed at ~39 ft). Allow MPC tracking lag of a few tenths of a second.
  assert r["steady_T_eq"] is not None
  assert r["steady_T_eq"] == pytest.approx(1.25, abs=0.25)
  assert r["steady_gap_ft"] == pytest.approx(desired_30 / FT, abs=15.0)
  # Crossing 25/30 must not be the old close-in.
  for mph in ("25", "30"):
    snap = r["at_speed"][mph]
    assert snap["gap_ft"] >= 50.0, snap
    assert snap["below_floor"] is False
    assert snap["launch_active"] is False
  # Take-up is still early (the comfortable first step is kept).
  assert r["t_move"] is not None and r["t_move"] <= 0.60
  # Launch must actually release.
  assert r["launch_on_at_end"] is False
  assert r["launch_first_off_s"] is not None
  assert r["launch_first_off_s"] < 20.0


def test_relaxed_launch_releases_and_does_not_close_in(kw):
  from openpilot.sunnypilot.selfdrive.controls.tests.test3_launchfix_sim import (
    run_launch_follow,
  )
  P = kw["P"]
  r = run_launch_follow(P.relaxed, kw, target_mph=30, sim_t=40.0)
  assert r["floor_ft"] == pytest.approx(29.69, abs=0.02)
  assert r["below_floor"] is False
  assert r["min_gap_ft"] >= 29.6
  # Pre-fix Relaxed stayed launch-active the entire 40 s at ~38.8 ft / 0.89 s.
  assert r["launch_on_at_end"] is False
  assert r["launch_first_off_s"] is not None
  assert r["launch_first_off_s"] < 20.0
  # Relaxed T at 30 is 1.75; desired ≈ 96.7 ft. Must not stay at ~39 ft.
  desired_30 = 1.75 * 30 * MPH + 6.0
  assert r["steady_gap_ft"] == pytest.approx(desired_30 / FT, abs=20.0)
  assert r["steady_T_eq"] == pytest.approx(1.75, abs=0.30)
  snap = r["at_speed"]["30"]
  assert snap["gap_ft"] >= 50.0
  assert snap["launch_active"] is False
