from fvhoe.initial_conditions import square, sinus
from fvhoe.fv import fv_average
import numpy as np
import pytest
from tests.utils import mse
from typing import Tuple


def meshgen(
    n: Tuple[float, float, float],
    x: Tuple[float, float] = (0, 1),
    y: Tuple[float, float] = (0, 1),
    z: Tuple[float, float] = (0, 1),
) -> Tuple[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    args:
        n (Tuple[float, float, float]) : number of cells (nx, ny, nz)
        x (Tuple[float, float]) : x boundaries (x0, x1)
        y (Tuple[float, float]) : y boundaries (y0, y1)
        z (Tuple[float, float]) : z boundaries (z0, z1)
    returns:
        X, Y, Z (Tuple[array_like, array_like, array_like]) : mesh arrays
        hx, hy, hz (Tuple[float, float, float]) : mesh spacings
    """
    # define cell boundaries
    xi = np.linspace(x[0], x[1], n[0] + 1)
    yi = np.linspace(y[0], y[1], n[1] + 1)
    zi = np.linspace(z[0], z[1], n[2] + 1)
    # define cell centers
    x, y, z = (
        0.5 * (xi[1:] + xi[:-1]),
        0.5 * (yi[1:] + yi[:-1]),
        0.5 * (zi[1:] + zi[:-1]),
    )
    # mesh
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    hx, hy, hz = (x[1] - x[0]) / n[0], (y[1] - y[0]) / n[1], (z[1] - z[0]) / n[2]
    return (X, Y, Z), (hx, hy, hz)


def test_first_order_cell_average():
    """
    assert first-order finite volume average should is trivial
    """
    (X, Y, Z), h = meshgen((32, 64, 128))

    def f(x, y, z):
        return sinus(x, y, z, dims="xyz", vx=1, vy=2, vz=3)

    assert np.all(f(X, Y, Z) == fv_average(f=f, x=X, y=Y, z=Z, h=h))


@pytest.mark.parametrize("px", [0, 1, 2, 3])
@pytest.mark.parametrize("py", [0, 1, 2, 3])
@pytest.mark.parametrize("pz", [0, 1, 2, 3])
def test_uniform_cell_average(px, py, pz):
    """
    assert finite volume average of a uniform region is trivial
    args:
        px (int) : polynomial interpolation degree in x-direction
        py (int) : polynomial interpolation degree in y-direction
        pz (int) : polynomial interpolation degree in z-direction
    """
    (X, Y, Z), h = meshgen((32, 64, 128))

    def f(x, y, z):
        return square(x, y, z, dims="xyz", vx=1, vy=2, vz=3)

    assert mse(f(X, Y, Z), fv_average(f=f, x=X, y=Y, z=Z, h=h, p=(px, py, pz))) < 1e-15
