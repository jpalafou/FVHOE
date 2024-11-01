from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.hydro import HydroState
from fvhoe.initial_conditions import Shock1D
from fvhoe.solver import EulerSolver
import numpy as np
import pytest

_hs = HydroState(ndim=1)


@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_passive_scalars(dim: str, N: int = 64, p: int = 3):
    """
    Test that including passive scalars doesn't change the solution
    args:
        dim (str): dimension of the test
        N (int): number of cells
        p (int): polynomial degree
    """
    # set up solvers
    shared_solver_config = dict(
        w0=Shock1D(dim=dim),
        bc=BoundaryCondition(x="free"),
        gamma=1.4,
        nx=N if dim == "x" else 1,
        ny=N if dim == "y" else 1,
        nz=N if dim == "z" else 1,
        px=p if dim == "x" else 0,
        py=p if dim == "y" else 0,
        pz=p if dim == "z" else 0,
        riemann_solver="hllc",
        a_posteriori_slope_limiting=p > 0,
        NAD=1e-5,
    )
    solver = EulerSolver(**shared_solver_config)
    solver_with_passives = EulerSolver(
        w0_passives={
            "tracer1": lambda x, y, z: np.sin(2 * np.sin(x)),
            "tracer2": lambda x, y, z: np.cos(2 * np.sin(x)),
        },
        **shared_solver_config,
    )

    # run solvers
    solver.run(0.245)
    solver_with_passives.run(0.245)

    # assert that the solutions are the same
    assert np.all(
        solver.snapshots[-1]["w"][_hs("active_scalars")]
        == solver_with_passives.snapshots[-1]["w"][_hs("active_scalars")]
    )
