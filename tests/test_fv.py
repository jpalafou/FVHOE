from fvhoe.fv import fv_average
import numpy as np
import pytest
from tests.utils import mse
from typing import Tuple


def sinus(x, y, z):
    out = np.empty((2, *x.shape))
    out[0] = 0.5 * np.sin(2 * np.pi * (x + y + z)) + 1.5
    out[1] = 0.5 * np.sin(2 * np.pi * (x + y + z)) + 2.5
    return out


def square(x, y, z):
    out = np.empty((2, *x.shape))
    inside_x_square = np.logical_and(x > 0.25, x < 0.75)
    inside_y_square = np.logical_and(y > 0.25, y < 0.75)
    inside_z_square = np.logical_and(z > 0.25, z < 0.75)
    inside_square = np.logical_and(
        np.logical_and(inside_x_square, inside_y_square), inside_z_square
    )
    out[0] = np.where(inside_square, 2, 1)
    out[1] = np.where(inside_square, 3, 2)
    return out


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
    a first-order finite volume average should be trivial
    """
    (X, Y, Z), h = meshgen((32, 64, 128))
    f = sinus
    assert np.all(f(X, Y, Z) == fv_average(f=f, x=X, y=Y, z=Z, h=h))


@pytest.mark.parametrize("px", [0, 1, 2, 3])
@pytest.mark.parametrize("py", [0, 1, 2, 3])
@pytest.mark.parametrize("pz", [0, 1, 2, 3])
def test_uniform_cell_average(px, py, pz):
    """
    the finite volume average of a uniform region shoudl be trivial
    """
    (X, Y, Z), h = meshgen((32, 64, 128))
    f = square
    assert mse(f(X, Y, Z), fv_average(f=f, x=X, y=Y, z=Z, h=h, p=(px, py, pz))) < 1e-15
