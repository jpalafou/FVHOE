from fvhoe.hydro import advection_dt, HydroState
from fvhoe.initial_conditions import Sinus, Square
from fvhoe.solver import EulerSolver
import pytest
from tests.utils import l2err


_hs = HydroState()


@pytest.mark.parametrize("f0", [Sinus, Square])
@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_1d_advection_symmetry(f0: callable, p: int, N: int = 64, t: float = 1):
    """
    assert symmetry of 3D solver along 3 directions
    args:
        f0 (callable) : initial condition function
        p (int) : polynomial degree along axis of interest
        N (int) : 1D resolution
        t (float) : time to solve up until
    """
    solutions = {}
    for dim in ["x", "y", "z"]:
        nx = {"x": N, "y": 1, "z": 1}[dim]
        ny = {"x": 1, "y": N, "z": 1}[dim]
        nz = {"x": 1, "y": 1, "z": N}[dim]
        px = {"x": p, "y": 0, "z": 0}[dim]
        py = {"x": 0, "y": p, "z": 0}[dim]
        pz = {"x": 0, "y": 0, "z": p}[dim]
        solver = EulerSolver(
            w0=f0(
                dims=dim,
                vx=1 if dim == "x" else 0,
                vy=1 if dim == "y" else 0,
                vz=1 if dim == "z" else 0,
            ),
            nx=nx,
            ny=ny,
            nz=nz,
            px=px,
            py=py,
            pz=pz,
            riemann_solver="advection_upwind",
            progress_bar=False,
            fixed_dt=0.4 * advection_dt(h=1 / N, vx=1),
        )
        solver.run(t)
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


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_2d_advection_symmetry(p, N=32, t: float = 1):
    """
    assert symmetry of 3D solver along 3 planes
    args:
        p (int) : polynomial interpolation degree along axes of interest
        N (int) : 2D resolution
        t (float) : time to solve up until
    """
    solutions = {}
    for dims in ["xy", "yz", "zx"]:
        nx = {"xy": N, "yz": 1, "zx": N}[dims]
        ny = {"xy": N, "yz": N, "zx": 1}[dims]
        nz = {"xy": 1, "yz": N, "zx": N}[dims]
        px = {"xy": p, "yz": 0, "zx": p}[dims]
        py = {"xy": p, "yz": p, "zx": 0}[dims]
        pz = {"xy": 0, "yz": p, "zx": p}[dims]
        solver = EulerSolver(
            w0=Square(
                dims=dims,
                vx={"xy": 2, "yz": 0, "zx": 2}[dims],
                vy={"xy": 1, "yz": 2, "zx": 0}[dims],
                vz={"xy": 0, "yz": 1, "zx": 1}[dims],
            ),
            nx=nx,
            ny=ny,
            nz=nz,
            px=px,
            py=py,
            pz=pz,
            riemann_solver="advection_upwind",
            progress_bar=False,
            fixed_dt=0.4 * advection_dt(h=1 / N, vx=2, vy=1),
        )
        solver.run(t)
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
