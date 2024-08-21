from functools import partial
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import double_shock_1d, sedov
from fvhoe.solver import EulerSolver
import pytest
from tests.test_utils import l2err


@pytest.mark.parametrize("p", [8])
@pytest.mark.parametrize("rs", ["llf", "hllc"])
@pytest.mark.parametrize(
    "limiting_config", [dict(), dict(SED=True), dict(SED=True, convex=True)]
)
def test_1d_symmetry(p: int, rs: str, limiting_config: dict, N: int = 100):
    """
    use the double shock 1D problem to test that the 1D solver is symmetric in all dimensions
    args:
        p (int) : polynomial degree along axis of interest
        rs (str) : riemann solver
        limiting_config (dict) : limiting configurations
        N (int) : 1D resolution
    """

    solutions = {}
    for dim in ["x", "y", "z"]:
        solver = EulerSolver(
            w0=partial(double_shock_1d, dim=dim),
            nx=N if dim == "x" else 1,
            ny=N if dim == "y" else 1,
            nz=N if dim == "z" else 1,
            px=p if dim == "x" else 0,
            py=p if dim == "y" else 0,
            pz=p if dim == "z" else 0,
            riemann_solver=rs,
            bc=BoundaryCondition(
                x="reflective" if dim == "x" else "periodic",
                y="reflective" if dim == "y" else "periodic",
                z="reflective" if dim == "z" else "periodic",
            ),
            gamma=1.4,
            all_floors=True,
            a_posteriori_slope_limiting=True,
            **limiting_config,
        )
        solver.rkorder(0.038)
        solutions[dim] = solver

    xyerr = l2err(
        solutions["x"].snapshots[-1]["w"].rho[:, 0, 0],
        solutions["y"].snapshots[-1]["w"].rho[0, :, 0],
    )
    yzerr = l2err(
        solutions["y"].snapshots[-1]["w"].rho[0, :, 0],
        solutions["z"].snapshots[-1]["w"].rho[0, 0, :],
    )

    assert xyerr == 0
    assert yzerr == 0


@pytest.mark.parametrize("p", [8])
@pytest.mark.parametrize("rs", ["llf", "hllc"])
@pytest.mark.parametrize(
    "limiting_config", [dict(), dict(SED=True), dict(SED=True, convex=True)]
)
def test_2d_symmetry(p: int, rs: str, limiting_config: dict, N: int = 32):
    """
    use the 2D sedov blast problem to test that the 2D solver is symmetric in all dimensions
    args:
        p (int) : polynomial degree along axis of interest
        rs (str) : riemann solver
        limiting_config (dict) : limiting configurations
        N (int) : 2D resolution
    """

    solutions = {}
    for dims in ["xy", "yz", "zx"]:
        solver = EulerSolver(
            w0=partial(sedov, dims=dims),
            conservative_ic=True,
            fv_ic=True,
            nx=N if "x" in dims else 1,
            ny=N if "y" in dims else 1,
            nz=N if "z" in dims else 1,
            px=p if "x" in dims else 0,
            py=p if "y" in dims else 0,
            pz=p if "z" in dims else 0,
            riemann_solver=rs,
            bc=BoundaryCondition(
                x=("reflective", "outflow") if "x" in dims else None,
                y=("reflective", "outflow") if "y" in dims else None,
                z=("reflective", "outflow") if "z" in dims else None,
            ),
            gamma=1.4,
            all_floors=True,
            a_posteriori_slope_limiting=True,
            **limiting_config,
        )
        solver.rkorder(0.3)
        solutions[dims] = solver

    xy_yz_err = l2err(
        solutions["xy"].snapshots[-1]["w"].rho[:, :, 0],
        solutions["yz"].snapshots[-1]["w"].rho[0, :, :],
    )
    yz_zx_err = l2err(
        solutions["yz"].snapshots[-1]["w"].rho[0, :, :],
        solutions["zx"].snapshots[-1]["w"].rho[:, 0, :],
    )

    assert xy_yz_err == 0
    assert yz_zx_err == 0
