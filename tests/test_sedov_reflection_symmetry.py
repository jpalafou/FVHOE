from functools import partial
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import shock_tube
from fvhoe.solver import EulerSolver
import numpy as np
import pytest
from tests.test_utils import l1err


@pytest.mark.parametrize("dims", ["xy", "yz", "zx", "xyz"])
@pytest.mark.parametrize("rs", ["llf", "hllc"])
@pytest.mark.parametrize("p", [0, 1])
def test_reflection_symmetry(dims: str, rs: str, p: int, N: int = 16, t: float = 0.8):
    """
    reflective boundary conditions should give the same result as a full domain. also check that the solution is symmetric
    args:
        dims (str) : dimensions to test
        rs (str) : riemann solver to use
        p (int) : polynomial interpolation degree
        N (int) : number of cells in each dimension
        t (float) : time at solution
    """
    # sedov blast problem in 2D
    rho0 = 1
    E0 = 1
    gamma = 1.4

    sedov_2d_configs = dict(
        gamma=gamma,
        CFL=0.8,
        px=p if "x" in dims else 0,
        py=p if "y" in dims else 0,
        pz=p if "z" in dims else 0,
        riemann_solver=rs,
        a_posteriori_slope_limiting=True,
        force_trouble=p > 0,
        slope_limiter="minmod",
        all_floors=True,
        conservative_ic=True,
        fv_ic=True,
    )

    # set up solvers
    solver_partial = EulerSolver(
        w0=partial(
            shock_tube,
            mode="cube",
            x_cube=(0, 1 / N) if "x" in dims else None,
            y_cube=(0, 1 / N) if "y" in dims else None,
            z_cube=(0, 1 / N) if "z" in dims else None,
            rho_in_out=(rho0, rho0),
            P_in_out=(0.25 * (N**2) * E0, 1e-5),
            conservative=True,
        ),
        bc=BoundaryCondition(
            x=("reflective", "outflow") if "x" in dims else None,
            y=("reflective", "outflow") if "y" in dims else None,
            z=("reflective", "outflow") if "z" in dims else None,
        ),
        nx=N if "x" in dims else 1,
        ny=N if "y" in dims else 1,
        nz=N if "z" in dims else 1,
        **sedov_2d_configs,
    )
    solver_full = EulerSolver(
        w0=partial(
            shock_tube,
            mode="cube",
            x_cube=(-1 / N, 1 / N) if "x" in dims else None,
            y_cube=(-1 / N, 1 / N) if "y" in dims else None,
            z_cube=(-1 / N, 1 / N) if "z" in dims else None,
            rho_in_out=(rho0, rho0),
            P_in_out=(0.25 * (N**2) * E0, 1e-5),
            conservative=True,
        ),
        x=(-1, 1) if "x" in dims else (0, 1),
        y=(-1, 1) if "y" in dims else (0, 1),
        z=(-1, 1) if "z" in dims else (0, 1),
        bc=BoundaryCondition(x="outflow", y="outflow", z="outflow"),
        nx=2 * N if "x" in dims else 1,
        ny=2 * N if "y" in dims else 1,
        nz=2 * N if "z" in dims else 1,
        **sedov_2d_configs,
    )

    # run solvers
    solver_partial.rkorder(t)
    solver_full.rkorder(t)

    # check results
    w_partial = solver_partial.snapshots[-1]["w"]
    w_full = solver_full.snapshots[-1]["w"][
        :,
        slice(N, None) if "x" in dims else slice(None),
        slice(N, None) if "y" in dims else slice(None),
        slice(N, None) if "z" in dims else slice(None),
    ]

    # check reflection equivalence
    assert l1err(w_partial.P, w_full.P) < 1e-15

    # check symmetry of partial solution
    if "x" in dims and "y" in dims:
        assert l1err(w_partial.P, np.swapaxes(w_partial.P, 0, 1)) < 1e-15
    if "y" in dims and "z" in dims:
        assert l1err(w_partial.P, np.swapaxes(w_partial.P, 1, 2)) < 1e-15
    if "z" in dims and "x" in dims:
        assert l1err(w_partial.P, np.swapaxes(w_partial.P, 2, 0)) < 1e-15
