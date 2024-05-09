from fvhoe.initial_conditions import sinus, square
from fvhoe.solver import EulerSolver
import pytest
from tests.utils import mse


@pytest.mark.parametrize("f0", [sinus, square])
@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_1d_advection_symmetry(f0, p):
    solutions = {}
    for dir in ["x", "y", "z"]:

        def directional_f0(x, y, z):
            vx = {"x": 1, "y": 0, "z": 0}[dir]
            vy = {"x": 0, "y": 1, "z": 0}[dir]
            vz = {"x": 0, "y": 0, "z": 1}[dir]
            return f0(x, y, z, dims=dir, vx=vx, vy=vy, vz=vz)

        nx = {"x": 64, "y": 1, "z": 1}[dir]
        ny = {"x": 1, "y": 64, "z": 1}[dir]
        nz = {"x": 1, "y": 1, "z": 64}[dir]
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
        )
        solver.rkorder(stopping_time=1)
        solutions[dir] = solver

    xy_symmetry = (
        mse(
            solutions["x"].snapshots[1]["rho"][:, 0, 0],
            solutions["y"].snapshots[1]["rho"][0, :, 0],
        )
        == 0
    )
    yz_symmetry = (
        mse(
            solutions["y"].snapshots[1]["rho"][0, :, 0],
            solutions["z"].snapshots[1]["rho"][0, 0, :],
        )
        == 0
    )

    assert xy_symmetry and yz_symmetry
