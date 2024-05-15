import numpy as np
from typing import Tuple


def l1err(x: np.ndarray, y: np.ndarray) -> float:
    """
    args:
        x (array_like)
        y (array_like)
    returns:
        out (float) : L1 error between x and y
    """
    return np.mean(np.abs(x - y))


def l2err(x: np.ndarray, y: np.ndarray) -> float:
    """
    args:
        x (array_like)
        y (array_like)
    returns:
        out (float) : L2 error between x and y
    """
    return np.sqrt(np.mean(np.square(x - y)))


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
    xx, yy, zz = (
        0.5 * (xi[1:] + xi[:-1]),
        0.5 * (yi[1:] + yi[:-1]),
        0.5 * (zi[1:] + zi[:-1]),
    )
    # mesh
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing="ij")
    hx, hy, hz = (x[1] - x[0]) / n[0], (y[1] - y[0]) / n[1], (z[1] - z[0]) / n[2]
    return (X, Y, Z), (hx, hy, hz)
