from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import shock_tube_1d
from fvhoe.solver import EulerSolver
import numpy as np
import pickle


def test_save_load():
    # set up a simple 1D Euler problem
    N = 60
    p = 0
    solver = EulerSolver(
        w0=shock_tube_1d,
        nx=N,
        px=p,
        riemann_solver="llf",
        bc=BoundaryCondition(x="free"),
    )

    # solve and save snapshots
    solver.euler(0.245, filename="test", overwrite=True)

    # load snapshots
    snapshots = pickle.load(open("snapshots/test/arrs.pkl", "rb"))

    assert np.all(solver.snapshots[-1]["w"] == snapshots[-1]["w"])
