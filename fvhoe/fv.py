from functools import partial
import numpy as np
from typing import Tuple


def get_window(
    ndim: int, axis: int, cut: Tuple[int, int] = (0, 0), step: int = 1
) -> np.ndarray:
    """
    grab window of array along axis
    args:
        ndim (int) : number of axis
        axis (int) : along which to apply boundaries
        cut (Tuple[int, int]) : (# elems to remove from left, '' right)
        step (int) : step length
    returns:
        out (array_like) : slices specifying window
    """
    slices = [slice(None)] * ndim
    slices[axis] = slice(cut[0] or None, -cut[1] or None, step)
    out = tuple(slices)
    return out


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

    gw = partial(get_window, ndim=fvarr.ndim, axis=axis)

    if pos == "r":
        return conservative_interpolation(
            fvarr=fvarr[gw(step=-1)],
            p=p,
            axis=axis,
            pos="l",
        )[gw(step=-1)]

    match p:
        case 0:
            out = fvarr.copy()
        case 1:
            if pos == "l":
                out = (
                    1 * fvarr[gw(cut=(0, 2))]
                    + 4 * fvarr[gw(cut=(1, 1))]
                    + -1 * fvarr[gw(cut=(2, 0))]
                ) / 4
            elif pos == "c":
                out = (
                    0 * fvarr[gw(cut=(0, 2))]
                    + 1 * fvarr[gw(cut=(1, 1))]
                    + 0 * fvarr[gw(cut=(2, 0))]
                ) / 1
        case 2:
            if pos == "l":
                out = (
                    2 * fvarr[gw(cut=(0, 2))]
                    + 5 * fvarr[gw(cut=(1, 1))]
                    + -1 * fvarr[gw(cut=(2, 0))]
                ) / 6
            elif pos == "c":
                out = (
                    -1 * fvarr[gw(cut=(0, 2))]
                    + 26 * fvarr[gw(cut=(1, 1))]
                    + -1 * fvarr[gw(cut=(2, 0))]
                ) / 24
        case 3:
            if pos == "l":
                out = (
                    -1 * fvarr[gw(cut=(0, 4))]
                    + 10 * fvarr[gw(cut=(1, 3))]
                    + 20 * fvarr[gw(cut=(2, 2))]
                    + -6 * fvarr[gw(cut=(3, 1))]
                    + 1 * fvarr[gw(cut=(4, 0))]
                ) / 24
            elif pos == "c":
                out = (
                    0 * fvarr[gw(cut=(0, 4))]
                    + -1 * fvarr[gw(cut=(1, 3))]
                    + 26 * fvarr[gw(cut=(2, 2))]
                    + -1 * fvarr[gw(cut=(3, 1))]
                    + 0 * fvarr[gw(cut=(4, 0))]
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

    gw = partial(get_window, ndim=u.ndim, axis=axis)

    match p:
        case 0:
            out = u.copy()
        case 1:
            out = (
                0 * u[gw(cut=(0, 2))] + 1 * u[gw(cut=(1, 1))] + 0 * u[gw(cut=(2, 0))]
            ) / 1
        case 2:
            out = (
                1 * u[gw(cut=(0, 2))] + 22 * u[gw(cut=(1, 1))] + 1 * u[gw(cut=(2, 0))]
            ) / 24
        case 3:
            out = (
                0 * u[gw(cut=(0, 4))]
                + 1 * u[gw(cut=(1, 3))]
                + 22 * u[gw(cut=(2, 2))]
                + 1 * u[gw(cut=(3, 1))]
                + 0 * u[gw(cut=(4, 0))]
            ) / 24
        case _:
            raise NotImplementedError(f"{p=}")

    return out


def fv_average(
    f: callable,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    h: Tuple[float, float, float],
    p: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    compute finite volume average of f over 3D domain
    args:
        f (callable) : f(x, y, z) -> array_like of shape (n, nx, ny, nz)
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        h (Tuple[float, float, float]) : grid spacing (hx, hy, hz)
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
