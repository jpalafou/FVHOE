from fvhoe.initial_conditions import sinus, square
from fvhoe.solver import EulerSolver
import pytest
from tests.test_utils import mse


@pytest.mark.parametrize("f0", [sinus, square])
@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_1d_advection_symmetry(f0: callable, p: int, N: int = 64):
    """
    assert symmetry of 3D solver along 3 directions
    args:
        f0 (callable) : initial condition function
        p (int) : polynomial degree along axis of interest
        N (int) : 1D resolution
    """
    solutions = {}
    for dir in ["x", "y", "z"]:

        def directional_f0(x, y, z):
            vx = {"x": 1, "y": 0, "z": 0}[dir]
            vy = {"x": 0, "y": 1, "z": 0}[dir]
            vz = {"x": 0, "y": 0, "z": 1}[dir]
            return f0(x, y, z, dims=dir, vx=vx, vy=vy, vz=vz)

        nx = {"x": N, "y": 1, "z": 1}[dir]
        ny = {"x": 1, "y": N, "z": 1}[dir]
        nz = {"x": 1, "y": 1, "z": N}[dir]
        px = {"x": p, "y": 0, "z": 0}[dir]
        py = {"x": 0, "y": p, "z": 0}[dir]
        pz = {"x": 0, "y": 0, "z": p}[dir]
        solver = EulerSolver(
            w0=directional_f0,
            nx=nx,
            ny=ny,
            nz=nz,
            px=px,
            py=py,
            pz=pz,
            riemann_solver="advection_upwind",
            progress_bar=False,
        )
        solver.rkorder(stopping_time=1)
        solutions[dir] = solver

    assert (
        mse(
            solutions["x"].snapshots[1]["rho"][:, 0, 0],
            solutions["y"].snapshots[1]["rho"][0, :, 0],
        )
        < 1e-20
    )

    assert (
        mse(
            solutions["y"].snapshots[1]["rho"][0, :, 0],
            solutions["z"].snapshots[1]["rho"][0, 0, :],
        )
        < 1e-20
    )


@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_2d_advection_symmetry(p, N=32):
    """
    assert symmetry of 3D solver along 3 planes
    args:
        p (int) : polynomial interpolation degree along axes of interest
        N (int) : 2D resolution
    """
    solutions = {}
    for dims in ["xy", "yz", "zx"]:

        def directional_f0(x, y, z):
            vx = {"xy": 2, "yz": 0, "zx": 2}[dims]
            vy = {"xy": 1, "yz": 2, "zx": 0}[dims]
            vz = {"xy": 0, "yz": 1, "zx": 1}[dims]
            return square(x, y, z, dims=dims, vx=vx, vy=vy, vz=vz)

        nx = {"xy": N, "yz": 1, "zx": N}[dims]
        ny = {"xy": N, "yz": N, "zx": 1}[dims]
        nz = {"xy": 1, "yz": N, "zx": N}[dims]
        px = {"xy": p, "yz": 0, "zx": p}[dims]
        py = {"xy": p, "yz": p, "zx": 0}[dims]
        pz = {"xy": 0, "yz": p, "zx": p}[dims]
        solver = EulerSolver(
            w0=directional_f0,
            nx=nx,
            ny=ny,
            nz=nz,
            px=px,
            py=py,
            pz=pz,
            riemann_solver="advection_upwind",
            progress_bar=False,
        )
        solver.rkorder(stopping_time=1)
        solutions[dims] = solver

    assert (
        mse(
            solutions["xy"].snapshots[1]["rho"][:, :, 0],
            solutions["yz"].snapshots[1]["rho"][0, :, :],
        )
        < 1e-20
    )

    assert (
        mse(
            solutions["yz"].snapshots[1]["rho"][0, :, :],
            solutions["zx"].snapshots[1]["rho"][:, 0, :],
        )
        < 1e-20
    )
