from functools import partial
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import shock_tube_2d
from fvhoe.solver import EulerSolver
import numpy as np
import pytest


@pytest.mark.parametrize("N", [64])
@pytest.mark.parametrize("p", [7])
@pytest.mark.parametrize("NAD", [0, 1e-3, 1e-5])
def test_NAD(N: int, p: int, NAD: float):
    """
    Test equivalence of NAD implementations for modes "any" and "only".
    args:
        N:      number of cells in each dimension
        p:      polynomial degree
        NAD:    NAD tolerance
    """
    # initialize solvers
    solver_configs = dict(
        w0=partial(shock_tube_2d, radius=0.05, rho_in_out=(1, 1), P_in_out=(1, 1e-5)),
        bc=BoundaryCondition(x="free", y="free"),
        CFL=0.8,
        nx=N,
        ny=N,
        px=p,
        py=p,
        riemann_solver="hllc",
        a_posteriori_slope_limiting=True,
        all_floors=True,
        NAD=NAD,
    )
    solver_any = EulerSolver(**solver_configs, NAD_mode="any")
    solver_only = EulerSolver(
        **solver_configs, NAD_mode="only", NAD_vars=["rho", "P", "vx", "vy"]
    )

    # run solvers
    solver_any.rkorder(0.05, save_snapshots=False)
    solver_only.rkorder(0.05, save_snapshots=False)

    # compare results
    assert np.all(solver_any.snapshots[-1]["w"] == solver_only.snapshots[-1]["w"])
