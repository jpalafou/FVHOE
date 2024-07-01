from functools import partial
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import double_shock_1d, shock_tube_2d
from fvhoe.solver import EulerSolver
import pytest
from tests.test_utils import l2err


@pytest.mark.parametrize("N", [100])
@pytest.mark.parametrize("p", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("convex", [False, True])
def test_1d_symmetry(N: int, p: int, convex: bool):
    """
    assert symmetry of 3D solver along 3 directions
    args:
        N (int) : 1D resolution
        p (int) : polynomial degree along axis of interest
        convex (bool) : convex limiting
        SED (bool) : use SED slope limiting
    """

    T = 0.038

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
            riemann_solver="hllc",
            bc=BoundaryCondition(
                x=(
                    {
                        "rho": "reflective",
                        "E": "reflective",
                        "mx": "negative-reflective",
                        "my": "periodic",
                        "mz": "periodic",
                    }
                    if dim == "x"
                    else "periodic"
                ),
                y=(
                    {
                        "rho": "reflective",
                        "E": "reflective",
                        "mx": "periodic",
                        "my": "negative-reflective",
                        "mz": "periodic",
                    }
                    if dim == "y"
                    else "periodic"
                ),
                z=(
                    {
                        "rho": "reflective",
                        "E": "reflective",
                        "mx": "periodic",
                        "my": "periodic",
                        "mz": "negative-reflective",
                    }
                    if dim == "z"
                    else "periodic"
                ),
            ),
            gamma=1.4,
            a_posteriori_slope_limiting=True,
            convex=convex,
            all_floors=True,
            slope_limiter="minmod",
        )
        solver.rkorder(T, save_snapshots=False)
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


@pytest.mark.parametrize("N", [32])
@pytest.mark.parametrize("p", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("convex", [False, True])
def test_2d_symmetry(N: int, p: int, convex: bool):
    T = 0.3
    P_blast = (1.4 - 1) * 0.25 * N**2

    solutions = {}
    for dims in ["xy", "yz", "zx"]:
        solver = EulerSolver(
            w0=partial(
                shock_tube_2d,
                dims=dims,
                center=(
                    0.5 / N if "x" in dims else 0,
                    0.5 / N if "y" in dims else 0,
                    0.5 / N if "z" in dims else 0,
                ),
                radius=0,
                rho_in_out=(1, 1),
                P_in_out=(P_blast, 1e-5),
            ),
            fv_ic=True,
            nx=N if "x" in dims else 1,
            ny=N if "y" in dims else 1,
            nz=N if "z" in dims else 1,
            px=p if "x" in dims else 0,
            py=p if "y" in dims else 0,
            pz=p if "z" in dims else 0,
            riemann_solver="llf",
            bc=BoundaryCondition(
                x=(
                    {
                        "rho": "reflective",
                        "mx": "negative-reflective",
                        "my": "reflective" if "y" in dims else "periodic",
                        "mz": "reflective" if "z" in dims else "periodic",
                        "E": "reflective",
                    }
                    if "x" in dims
                    else "periodic"
                ),
                y=(
                    {
                        "rho": "reflective",
                        "mx": "reflective" if "x" in dims else "periodic",
                        "my": "negative-reflective",
                        "mz": "reflective" if "z" in dims else "periodic",
                        "E": "reflective",
                    }
                    if "y" in dims
                    else "periodic"
                ),
                z=(
                    {
                        "rho": "reflective",
                        "mx": "reflective" if "x" in dims else "periodic",
                        "my": "reflective" if "y" in dims else "periodic",
                        "mz": "negative-reflective",
                        "E": "reflective",
                    }
                    if "z" in dims
                    else "periodic"
                ),
            ),
            gamma=1.4,
            a_posteriori_slope_limiting=True,
            convex=convex,
            all_floors=True,
            slope_limiter="minmod",
        )
        solver.rkorder(T, save_snapshots=False)
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
