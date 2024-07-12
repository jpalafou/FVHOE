from functools import partial
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import shock_tube
from fvhoe.solver import EulerSolver
import numpy as np
import pytest
from tests.test_utils import l1err


@pytest.mark.parametrize("dims", ["xy", "yz", "zx"])
@pytest.mark.parametrize("rs", ["llf", "hllc"])
@pytest.mark.parametrize("p", [0, 1])
def test_2D(dims: str, rs: str, p: int, N: int = 16, t: float = 0.8):
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
        NAD=1e-3,
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
    w_full = solver_full.snapshots[-1]["w"]
    slices = (
        slice(N, None) if "x" in dims else slice(None),
        slice(N, None) if "y" in dims else slice(None),
        slice(N, None) if "z" in dims else slice(None),
    )

    rho_err = l1err(w_partial.rho, w_full.rho[slices])
    P_err = l1err(w_partial.P, w_full.P[slices])
    vx_err = l1err(w_partial.vx, w_full.vx[slices])
    vy_err = l1err(w_partial.vy, w_full.vy[slices])
    vz_err = l1err(w_partial.vz, w_full.vz[slices])

    assert np.all(np.array([rho_err, P_err, vx_err, vy_err, vz_err]) < 1e-15)
