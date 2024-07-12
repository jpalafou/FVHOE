from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import shock_1d
from fvhoe.solver import EulerSolver
import numpy as np


def test_save_load():
    # set up a simple 1D Euler problem
    N = 60
    p = 0
    solver_config = dict(
        w0=shock_1d,
        nx=N,
        px=p,
        riemann_solver="llf",
        bc=BoundaryCondition(x="free"),
    )
    solver = EulerSolver(**solver_config)

    # solve and save snapshots
    solver.euler(0.245, snapshot_dir="snapshots/test", overwrite=True)

    # load snapshots
    solver2 = EulerSolver(**solver_config)
    solver2.euler(0.245, snapshot_dir="snapshots/test")

    assert np.all(solver.snapshots[-1]["w"] == solver2.snapshots[-1]["w"])
