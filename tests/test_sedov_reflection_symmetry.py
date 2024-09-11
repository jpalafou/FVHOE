from fvhoe.array_manager import get_array_slice as slc
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import sedov
from fvhoe.solver import EulerSolver
import numpy as np
import pytest
from tests.test_utils import l1err


@pytest.mark.parametrize("dims", ["xy", "yz", "zx", "xyz"])
@pytest.mark.parametrize("rs", ["llf", "hllc"])
@pytest.mark.parametrize("p", [0, 1])
def test_reflection_symmetry(dims: str, rs: str, p: int, N: int = 16):
    """
    reflective boundary conditions should give the same result as a full domain. also check that the solution is symmetric
    args:
        dims (str) : dimensions to test
        rs (str) : riemann solver to use
        p (int) : polynomial interpolation degree
        N (int) : number of cells in each dimension
    """

    sedov_configs = dict(
        gamma=1.4,
        CFL=0.8,
        px=p if "x" in dims else 0,
        py=p if "y" in dims else 0,
        pz=p if "z" in dims else 0,
        riemann_solver=rs,
        a_posteriori_slope_limiting=p > 0,
        force_trouble=p > 0,
        slope_limiter="minmod",
        density_floor=1e-16,
        pressure_floor=1e-16,
        conservative_ic=True,
        fv_ic=True,
    )

    # set up solvers
    solver_partial = EulerSolver(
        w0=sedov(dims=dims, mode="corner"),
        bc=BoundaryCondition(
            x=("reflective", "outflow") if "x" in dims else None,
            y=("reflective", "outflow") if "y" in dims else None,
            z=("reflective", "outflow") if "z" in dims else None,
        ),
        nx=N if "x" in dims else 1,
        ny=N if "y" in dims else 1,
        nz=N if "z" in dims else 1,
        **sedov_configs,
    )
    solver_full = EulerSolver(
        w0=sedov(dims=dims, mode="center"),
        x=(-1, 1) if "x" in dims else (0, 1),
        y=(-1, 1) if "y" in dims else (0, 1),
        z=(-1, 1) if "z" in dims else (0, 1),
        bc=BoundaryCondition(x="outflow", y="outflow", z="outflow"),
        nx=2 * N if "x" in dims else 1,
        ny=2 * N if "y" in dims else 1,
        nz=2 * N if "z" in dims else 1,
        **sedov_configs,
    )

    # run solvers
    solver_partial.rkorder(0.8)
    solver_full.rkorder(0.8)

    # check results
    w_partial = solver_partial.snapshots[-1]["w"]
    w_full = solver_full.snapshots[-1]["w"][
        :,
        slice(N, None) if "x" in dims else slice(None),
        slice(N, None) if "y" in dims else slice(None),
        slice(N, None) if "z" in dims else slice(None),
    ]

    # check reflection equivalence
    assert l1err(w_partial[slc("P")], w_full[slc("P")]) < 1e-15

    # check symmetry of partial solution
    if "x" in dims and "y" in dims:
        assert (
            l1err(w_partial[slc("P")], np.swapaxes(w_partial[slc("P")], 0, 1)) < 1e-15
        )
    if "y" in dims and "z" in dims:
        assert (
            l1err(w_partial[slc("P")], np.swapaxes(w_partial[slc("P")], 1, 2)) < 1e-15
        )
    if "z" in dims and "x" in dims:
        assert (
            l1err(w_partial[slc("P")], np.swapaxes(w_partial[slc("P")], 2, 0)) < 1e-15
        )
