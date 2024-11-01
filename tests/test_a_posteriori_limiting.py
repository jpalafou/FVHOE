from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.hydro import HydroState
from fvhoe.initial_conditions import DoubleShock1D, SedovBlast
from fvhoe.slope_limiting import detect_troubled_cells
from fvhoe.solver import EulerSolver
import numpy as np
import pytest
from tests.utils import l2err

_hs = HydroState(ndim=1)


@pytest.mark.parametrize("dim", ["x", "xy", "xyz"])
@pytest.mark.parametrize("NAD_eps", [1e-2, 1e-5])
@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("range_type", ["relative", "absolute"])
@pytest.mark.parametrize("NAD_vars", [None, ("rho", "P")])
@pytest.mark.parametrize("PAD_bounds", [None, {"rho": (0, np.inf), "P": (0, np.inf)}])
@pytest.mark.parametrize("SED", [False, True])
def test_detect_troubled_cells(
    dim: str,
    NAD_eps: float,
    mode: str,
    range_type: str,
    NAD_vars: list,
    PAD_bounds: dict,
    SED: bool,
    N: int = 64,
):
    """
    test the detect_troubled_cells function
    args:
        dim (str) : dimensionality of the problem
        NAD_eps (float) : NAD epsilon
        mode (str) : mode of detection
        range_type (str) : type of range
        NAD_vars (list) : NAD variables
        PAD_bounds (dict) : PAD bounds
        SED (bool) : SED
        N (int) : resolution
    """

    w = np.random.rand(
        5,
        N if "x" in dim else 1,
        N if "y" in dim else 1,
        N if "z" in dim else 1,
    )
    w_candidate = np.random.rand(
        5,
        N if "x" in dim else 1,
        N if "y" in dim else 1,
        N if "z" in dim else 1,
    )
    detect_troubled_cells(
        u=w,
        u_candidate=w_candidate,
        NAD_eps=NAD_eps,
        mode=mode,
        range_type=range_type,
        NAD_vars=NAD_vars,
        PAD_bounds=PAD_bounds,
        SED=SED,
    )


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
            w0=DoubleShock1D(dim=dim),
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
            density_floor=1e-16,
            pressure_floor=1e-16,
            a_posteriori_slope_limiting=True,
            **limiting_config,
        )
        solver.run(0.038)
        solutions[dim] = solver

    xyerr = l2err(
        solutions["x"].snapshots[-1]["w"][_hs("rho")][:, 0, 0],
        solutions["y"].snapshots[-1]["w"][_hs("rho")][0, :, 0],
    )
    yzerr = l2err(
        solutions["y"].snapshots[-1]["w"][_hs("rho")][0, :, 0],
        solutions["z"].snapshots[-1]["w"][_hs("rho")][0, 0, :],
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
            w0=SedovBlast(dims=dims),
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
            density_floor=1e-16,
            pressure_floor=1e-16,
            a_posteriori_slope_limiting=True,
            **limiting_config,
        )
        solver.run(0.3)
        solutions[dims] = solver

    xy_yz_err = l2err(
        solutions["xy"].snapshots[-1]["w"][_hs("rho")][:, :, 0],
        solutions["yz"].snapshots[-1]["w"][_hs("rho")][0, :, :],
    )
    yz_zx_err = l2err(
        solutions["yz"].snapshots[-1]["w"][_hs("rho")][0, :, :],
        solutions["zx"].snapshots[-1]["w"][_hs("rho")][:, 0, :],
    )

    assert xy_yz_err == 0
    assert yz_zx_err == 0
