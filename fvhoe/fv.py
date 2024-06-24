from functools import partial
import numpy as np
from typing import Tuple


def uniform_fv_mesh(
    nx: int,
    ny: int = 1,
    nz: int = 1,
    x: Tuple[float, float] = (0, 1),
    y: Tuple[float, float] = (0, 1),
    z: Tuple[float, float] = (0, 1),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    compute locations of finite volume centers in a uniform x, y, z mesh
    args:
        nx (int) : number of volumes in x-direction
        ny (int) : number of volumes in y-direction
        nz (int) : number of volumes in z-direction
        x (Tuple[float, float]) : left, right x-domain boundaries
        y (Tuple[float, float]) : left, right y-domain boundaries
        z (Tuple[float, float]) : left, right z-domain boundaries
    returns:
        X, Y, Z (Tuple[array_like, array_like, array_like]) : finite volume centers
    """
    x_interfaces = np.linspace(x[0], x[1], nx + 1)
    y_interfaces = np.linspace(y[0], y[1], ny + 1)
    z_interfaces = np.linspace(z[0], z[1], nz + 1)
    x_centers = 0.5 * (x_interfaces[:-1] + x_interfaces[1:])
    y_centers = 0.5 * (y_interfaces[:-1] + y_interfaces[1:])
    z_centers = 0.5 * (z_interfaces[:-1] + z_interfaces[1:])
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    return X, Y, Z


def get_view(
    ndim: int, axis: int, cut: Tuple[int, int] = (0, 0), step: int = 1
) -> tuple:
    """
    grab view of array along axis
    args:
        ndim (int) : number of axes
        axis (int) : along which to get view
        cut (Tuple[int, int]) : (# elems to remove from left, '' right)
        step (int) : step length
    returns:
        out (tuple) : series of slices specifying view
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
        pos (str) : interpolation position along finite volume
            "l" left
            "c" center
            "r" right
    returns:
        out (array_like) : array of interpolations
    """

    gv = partial(get_view, ndim=fvarr.ndim, axis=axis)

    if pos == "r":
        return conservative_interpolation(
            fvarr=fvarr[gv(step=-1)],
            p=p,
            axis=axis,
            pos="l",
        )[gv(step=-1)]

    match p:
        case 0:
            out = fvarr.copy()
        case 1:
            if pos == "l":
                out = (
                    1 * fvarr[gv(cut=(0, 2))]
                    + 4 * fvarr[gv(cut=(1, 1))]
                    + -1 * fvarr[gv(cut=(2, 0))]
                ) / 4
            elif pos == "c":
                out = (
                    0 * fvarr[gv(cut=(0, 2))]
                    + 1 * fvarr[gv(cut=(1, 1))]
                    + 0 * fvarr[gv(cut=(2, 0))]
                ) / 1
        case 2:
            if pos == "l":
                out = (
                    2 * fvarr[gv(cut=(0, 2))]
                    + 5 * fvarr[gv(cut=(1, 1))]
                    + -1 * fvarr[gv(cut=(2, 0))]
                ) / 6
            elif pos == "c":
                out = (
                    -1 * fvarr[gv(cut=(0, 2))]
                    + 26 * fvarr[gv(cut=(1, 1))]
                    + -1 * fvarr[gv(cut=(2, 0))]
                ) / 24
        case 3:
            if pos == "l":
                out = (
                    -1 * fvarr[gv(cut=(0, 4))]
                    + 10 * fvarr[gv(cut=(1, 3))]
                    + 20 * fvarr[gv(cut=(2, 2))]
                    + -6 * fvarr[gv(cut=(3, 1))]
                    + 1 * fvarr[gv(cut=(4, 0))]
                ) / 24
            elif pos == "c":
                out = (
                    0 * fvarr[gv(cut=(0, 4))]
                    + -1 * fvarr[gv(cut=(1, 3))]
                    + 26 * fvarr[gv(cut=(2, 2))]
                    + -1 * fvarr[gv(cut=(3, 1))]
                    + 0 * fvarr[gv(cut=(4, 0))]
                ) / 24
        case 7:
            if pos == "l":
                out = (
                    (-1 / 560) * fvarr[gv(cut=(0, 8))]
                    + (17 / 840) * fvarr[gv(cut=(1, 7))]
                    + (-97 / 840) * fvarr[gv(cut=(2, 6))]
                    + (449 / 840) * fvarr[gv(cut=(3, 5))]
                    + (319 / 420) * fvarr[gv(cut=(4, 4))]
                    + (-223 / 840) * fvarr[gv(cut=(5, 3))]
                    + (71 / 840) * fvarr[gv(cut=(6, 2))]
                    + (-1 / 56) * fvarr[gv(cut=(7, 1))]
                    + (1 / 560) * fvarr[gv(cut=(8, 0))]
                )
            elif pos == "c":
                out = (
                    0 * fvarr[gv(cut=(0, 8))]
                    + (-5 / 7168) * fvarr[gv(cut=(1, 7))]
                    + (159 / 17920) * fvarr[gv(cut=(2, 6))]
                    + (-7621 / 107520) * fvarr[gv(cut=(3, 5))]
                    + (30251 / 26880) * fvarr[gv(cut=(4, 4))]
                    + (-7621 / 107520) * fvarr[gv(cut=(5, 3))]
                    + (159 / 17920) * fvarr[gv(cut=(6, 2))]
                    + (-5 / 7168) * fvarr[gv(cut=(7, 1))]
                    + 0 * fvarr[gv(cut=(8, 0))]
                )
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

    gv = partial(get_view, ndim=u.ndim, axis=axis)

    match p:
        case 0:
            out = u.copy()
        case 1:
            out = (
                0 * u[gv(cut=(0, 2))] + 1 * u[gv(cut=(1, 1))] + 0 * u[gv(cut=(2, 0))]
            ) / 1
        case 2:
            out = (
                1 * u[gv(cut=(0, 2))] + 22 * u[gv(cut=(1, 1))] + 1 * u[gv(cut=(2, 0))]
            ) / 24
        case 3:
            out = (
                0 * u[gv(cut=(0, 4))]
                + 1 * u[gv(cut=(1, 3))]
                + 22 * u[gv(cut=(2, 2))]
                + 1 * u[gv(cut=(3, 1))]
                + 0 * u[gv(cut=(4, 0))]
            ) / 24
        case 7:
            out = (
                0 * u[gv(cut=(0, 8))]
                + (367 / 967680) * u[gv(cut=(1, 7))]
                + (-281 / 53760) * u[gv(cut=(2, 6))]
                + (6361 / 107520) * u[gv(cut=(3, 5))]
                + (215641 / 241920) * u[gv(cut=(4, 4))]
                + (6361 / 107520) * u[gv(cut=(5, 3))]
                + (-281 / 53760) * u[gv(cut=(6, 2))]
                + (367 / 967680) * u[gv(cut=(7, 1))]
                + 0 * u[gv(cut=(8, 0))]
            )
        case _:
            raise NotImplementedError(f"{p=}")

    return out


def interpolate_cell_centers(
    fvarr: np.ndarray, p: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    compute cell centers from an array of finite volume averages
    args:
        fvarr (array_like) : array of finite volume averages, with shape (# vars, nx, ny, nz)
        p (Tuple[int, int, int]) : polynomial degrees in each direction (px, py, pz)
    returns:
        out (array_like) : interpolations of cell centers
    """
    out = conservative_interpolation(
        fvarr=conservative_interpolation(
            fvarr=conservative_interpolation(fvarr=fvarr, p=p[0], axis=1, pos="c"),
            p=p[1],
            axis=2,
            pos="c",
        ),
        p=p[2],
        axis=3,
        pos="c",
    )
    return out


def interpolate_fv_averages(
    u: np.ndarray, p: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    compute finite volume averages from an array of cell centers
    args:
        u (array_like) : array of cell centers, with shape (# vars, nx, ny, nz)
        p (Tuple[int, int, int]) : polynomial degrees in each direction (px, py, pz)
    returns:
        out (array_like) : interpolations of cell centers
    """
    out = transverse_reconstruction(
        u=transverse_reconstruction(
            u=transverse_reconstruction(
                u=u,
                p=p[0],
                axis=1,
            ),
            p=p[1],
            axis=2,
        ),
        p=p[2],
        axis=3,
    )
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
