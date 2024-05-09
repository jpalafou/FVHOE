import numpy as np
from typing import Tuple


def conservative_interpolation(
    fvarr: np.ndarray, p: int, axis: int, pos: str = "c"
) -> np.ndarray:
    """
    args:
        fvarr (array_like) : array of finite volume cell averages
        p (int) : polynomial degree of conservative interpolation
        axis (int) : along which to interpolate
        pos (str) : cell position
            "l" left
            "c" center
            "r" right
    returns:
        out (array_like) : array of interpolations
    """

    def get_slices(x, end=(None, None), step=1):
        slices = [slice(None)] * fvarr.ndim
        slices[axis] = slice(end[0] or None, end[1] or None, step)
        return x[tuple(slices)]

    if pos == "r":
        return get_slices(
            conservative_interpolation(
                fvarr=get_slices(fvarr, step=-1), p=p, axis=axis, pos="l"
            ),
            step=-1,
        )

    match p:
        case 0:
            out = fvarr
        case 1:
            if pos == "l":
                out = (
                    1 * get_slices(fvarr, end=(0, -2))
                    + 4 * get_slices(fvarr, end=(1, -1))
                    + -1 * get_slices(fvarr, end=(2, 0))
                ) / 4
            elif pos == "c":
                out = (
                    0 * get_slices(fvarr, end=(0, -2))
                    + 1 * get_slices(fvarr, end=(1, -1))
                    + 0 * get_slices(fvarr, end=(2, 0))
                ) / 1
        case 2:
            if pos == "l":
                out = (
                    2 * get_slices(fvarr, end=(0, -2))
                    + 5 * get_slices(fvarr, end=(1, -1))
                    + -1 * get_slices(fvarr, end=(2, 0))
                ) / 6
            elif pos == "c":
                out = (
                    -1 * get_slices(fvarr, end=(0, -2))
                    + 26 * get_slices(fvarr, end=(1, -1))
                    + -1 * get_slices(fvarr, end=(2, 0))
                ) / 24
        case 3:
            if pos == "l":
                out = (
                    -1 * get_slices(fvarr, end=(0, -4))
                    + 10 * get_slices(fvarr, end=(1, -3))
                    + 20 * get_slices(fvarr, end=(2, -2))
                    + -6 * get_slices(fvarr, end=(3, -1))
                    + 1 * get_slices(fvarr, end=(4, 0))
                ) / 24
            elif pos == "c":
                out = (
                    0 * get_slices(fvarr, end=(0, -4))
                    + -1 * get_slices(fvarr, end=(1, -3))
                    + 26 * get_slices(fvarr, end=(2, -2))
                    + -1 * get_slices(fvarr, end=(3, -1))
                    + 0 * get_slices(fvarr, end=(4, 0))
                ) / 24
        case _:
            raise NotImplementedError(f"{p=}")

    return out


def transverse_reconstruction(u: np.ndarray, p: int, axis: int) -> np.ndarray:
    """
    args:
        u (array_like) : array of pointwise interpolations
        p (int) : polynomial degree of integral interpolation
        axis (int) : along which to interpolate
    returns:
        out (array_like) : array of interpolations of flux integrals
    """

    def get_slices(x, end=(None, None), step=1):
        slices = [slice(None)] * u.ndim
        slices[axis] = slice(end[0] or None, end[1] or None, step)
        return x[tuple(slices)]

    match p:
        case 0:
            out = u
        case 1:
            out = (
                0 * get_slices(u, end=(0, -2))
                + 1 * get_slices(u, end=(1, -1))
                + 0 * get_slices(u, end=(2, 0))
            ) / 1
        case 2:
            out = (
                -1 * get_slices(u, end=(0, -2))
                + 26 * get_slices(u, end=(1, -1))
                + -1 * get_slices(u, end=(2, 0))
            ) / 24
        case 3:
            out = (
                0 * get_slices(u, end=(0, -4))
                + -1 * get_slices(u, end=(1, -3))
                + 26 * get_slices(u, end=(2, -2))
                + -1 * get_slices(u, end=(3, -1))
                + 0 * get_slices(u, end=(4, 0))
            ) / 24
        case _:
            raise NotImplementedError(f"{p=}")

    return out


def fv_average(
    f: callable,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    h: Tuple[int, int, int],
    p: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    compute finite volume average of f over 3D domain
    args:
        f (callable) : f(x, y, z) -> array_like of shape (n, nx, ny, nz)
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        h (Tuple[int, int, int]) : grid spacing (hx, hy, hz)
        p (Tuple[int, int, int]) : interpolation polynomial degree (px, py, pz)
    returns:
        out (array_like) : finite volume averages of f(x, y, z), shape (n, nx, ny, nz)
    """
    # mesh spacings
    hx, hy, hz = h

    # quadrature points and weights
    points_and_weights = []
    for pi in p:
        points, weights = np.polynomial.legendre.leggauss(int(np.ceil((pi + 1) / 2)))
        points /= 2
        weights /= 2
        points_and_weights.append((points, weights))

    # find cell averages
    out = np.zeros_like(f(x, y, z))
    for xp, xw in zip(*points_and_weights[0]):
        for yp, yw in zip(*points_and_weights[1]):
            for zp, zw in zip(*points_and_weights[2]):
                weight = xw * yw * zw
                out += weight * f(x=x + xp * hx, y=y + yp * hy, z=z + zp * hz)

    return out
