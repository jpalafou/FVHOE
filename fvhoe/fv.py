from itertools import product
from functools import partial
import numpy as np
from typing import Tuple


def fv_uniform_meshgen(
    n: Tuple[float, float, float],
    x: Tuple[float, float] = (0, 1),
    y: Tuple[float, float] = (0, 1),
    z: Tuple[float, float] = (0, 1),
    slab_thickness: Tuple[int, int, int] = (0, 0, 0),
) -> tuple:
    """
    generate a 3D mesh of finite volume cell centers
    args:
        n (Tuple[float, float, float]) : number of cells (nx, ny, nz)
        x (Tuple[float, float]) : x boundaries (x0, x1)
        y (Tuple[float, float]) : y boundaries (y0, y1)
        z (Tuple[float, float]) : z boundaries (z0, z1)
    returns:
        if slab_thickness is (0, 0, 0):
            X, Y, Z (Tuple[array_like, array_like, array_like]) : mesh arrays, if slab_thickness is (0, 0, 0)
        if slab_thickness is not (0, 0, 0):
            inner_coords (Tuple[array_like, array_like, array_like]) : mesh arrays without slabs
            slab_coords (Dict[str, Tuple[array_like, array_like, array_like]]) : mesh arrays for slabs in each direction, indexed by dimension and position
            {'xl': (...), 'xr': (...), ...}
    """
    # define cell boundaries
    x_bound = np.linspace(x[0], x[1], n[0] + 1)
    y_bound = np.linspace(y[0], y[1], n[1] + 1)
    z_bound = np.linspace(z[0], z[1], n[2] + 1)

    # define cell centers
    x_mid, y_mid, z_mid = (
        0.5 * (x_bound[1:] + x_bound[:-1]),
        0.5 * (y_bound[1:] + y_bound[:-1]),
        0.5 * (z_bound[1:] + z_bound[:-1]),
    )

    # uniform mesh
    X, Y, Z = np.meshgrid(x_mid, y_mid, z_mid, indexing="ij")

    # early escape if slab coordinates are not needed
    if slab_thickness == (0, 0, 0):
        return X, Y, Z

    # store innter coordinates
    inner_coords = X, Y, Z

    # build dict of slab coordinates
    h = ((x[1] - x[0]) / n[0], (y[1] - y[0]) / n[1], (z[1] - z[0]) / n[2])
    gwx, gwy, gwz = slab_thickness
    slab_coords = {}
    for dim, pos in product("xyz", "lr"):
        X, Y, Z = fv_uniform_meshgen(
            (
                gwx if dim == "x" else n[0] + 2 * gwx,
                gwy if dim == "y" else n[1] + 2 * gwy,
                gwz if dim == "z" else n[2] + 2 * gwz,
            ),
            x=(
                {"l": (x[0] - h[0] * gwx, x[0]), "r": (x[1], x[1] + h[0] * gwx)}[pos]
                if dim == "x"
                else (x[0] - h[0] * gwx, x[1] + h[0] * gwx)
            ),
            y=(
                {"l": (y[0] - h[1] * gwy, y[0]), "r": (y[1], y[1] + h[1] * gwy)}[pos]
                if dim == "y"
                else (y[0] - h[1] * gwy, y[1] + h[1] * gwy)
            ),
            z=(
                {"l": (z[0] - h[2] * gwz, z[0]), "r": (z[1], z[1] + h[2] * gwz)}[pos]
                if dim == "z"
                else (z[0] - h[2] * gwz, z[1] + h[2] * gwz)
            ),
        )
        slab_coords[dim + pos] = (X, Y, Z)

    return inner_coords, slab_coords


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
        fvarr (array_like) : array of finite volume cell averages of arbitrary shape
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
        case 4:
            if pos == "l":
                out = (
                    -3 * fvarr[gv(cut=(0, 4))]
                    + 27 * fvarr[gv(cut=(1, 3))]
                    + 47 * fvarr[gv(cut=(2, 2))]
                    + -13 * fvarr[gv(cut=(3, 1))]
                    + 2 * fvarr[gv(cut=(4, 0))]
                ) / 60
            elif pos == "c":
                out = (
                    9 * fvarr[gv(cut=(0, 4))]
                    + -116 * fvarr[gv(cut=(1, 3))]
                    + 2134 * fvarr[gv(cut=(2, 2))]
                    + -116 * fvarr[gv(cut=(3, 1))]
                    + 9 * fvarr[gv(cut=(4, 0))]
                ) / 1920
        case 5:
            if pos == "l":
                out = (
                    (1 / 120) * fvarr[gv(cut=(0, 6))]
                    + (-1 / 12) * fvarr[gv(cut=(1, 5))]
                    + (59 / 120) * fvarr[gv(cut=(2, 4))]
                    + (47 / 60) * fvarr[gv(cut=(3, 3))]
                    + (-31 / 120) * fvarr[gv(cut=(4, 2))]
                    + (1 / 15) * fvarr[gv(cut=(5, 1))]
                    + (-1 / 120) * fvarr[gv(cut=(6, 0))]
                )
            elif pos == "c":
                out = (
                    0 * fvarr[gv(cut=(0, 6))]
                    + (3 / 640) * fvarr[gv(cut=(1, 5))]
                    + (-29 / 480) * fvarr[gv(cut=(2, 4))]
                    + (1067 / 960) * fvarr[gv(cut=(3, 3))]
                    + (-29 / 480) * fvarr[gv(cut=(4, 2))]
                    + (3 / 640) * fvarr[gv(cut=(5, 1))]
                    + 0 * fvarr[gv(cut=(6, 0))]
                )
        case 6:
            if pos == "l":
                out = (
                    (1 / 105) * fvarr[gv(cut=(0, 6))]
                    + (-19 / 210) * fvarr[gv(cut=(1, 5))]
                    + (107 / 210) * fvarr[gv(cut=(2, 4))]
                    + (319 / 420) * fvarr[gv(cut=(3, 3))]
                    + (-101 / 420) * fvarr[gv(cut=(4, 2))]
                    + (5 / 84) * fvarr[gv(cut=(5, 1))]
                    + (-1 / 140) * fvarr[gv(cut=(6, 0))]
                )
            elif pos == "c":
                out = (
                    (-5 / 7168) * fvarr[gv(cut=(0, 6))]
                    + (159 / 17920) * fvarr[gv(cut=(1, 5))]
                    + (-7621 / 107520) * fvarr[gv(cut=(2, 4))]
                    + (30251 / 26880) * fvarr[gv(cut=(3, 3))]
                    + (-7621 / 107520) * fvarr[gv(cut=(4, 2))]
                    + (159 / 17920) * fvarr[gv(cut=(5, 1))]
                    + (-5 / 7168) * fvarr[gv(cut=(6, 0))]
                )
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
        case 8:
            if pos == "l":
                out = (
                    (-1 / 504) * fvarr[gv(cut=(0, 8))]
                    + (11 / 504) * fvarr[gv(cut=(1, 7))]
                    + (-61 / 504) * fvarr[gv(cut=(2, 6))]
                    + (275 / 504) * fvarr[gv(cut=(3, 5))]
                    + (1879 / 2520) * fvarr[gv(cut=(4, 4))]
                    + (-641 / 2520) * fvarr[gv(cut=(5, 3))]
                    + (199 / 2520) * fvarr[gv(cut=(6, 2))]
                    + (-41 / 2520) * fvarr[gv(cut=(7, 1))]
                    + (1 / 630) * fvarr[gv(cut=(8, 0))]
                )
            elif pos == "c":
                out = (
                    (35 / 294912) * fvarr[gv(cut=(0, 8))]
                    + (-425 / 258048) * fvarr[gv(cut=(1, 7))]
                    + (31471 / 2580480) * fvarr[gv(cut=(2, 6))]
                    + (-100027 / 1290240) * fvarr[gv(cut=(3, 5))]
                    + (5851067 / 5160960) * fvarr[gv(cut=(4, 4))]
                    + (-100027 / 1290240) * fvarr[gv(cut=(5, 3))]
                    + (31471 / 2580480) * fvarr[gv(cut=(6, 2))]
                    + (-425 / 258048) * fvarr[gv(cut=(7, 1))]
                    + (35 / 294912) * fvarr[gv(cut=(8, 0))]
                )
        case _:
            raise NotImplementedError(f"{p=}")

    return out


def transverse_reconstruction(u: np.ndarray, p: int, axis: int) -> np.ndarray:
    """
    args:
        u (array_like) : array of pointwise interpolations of arbitrary shape
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
        case 4:
            out = (
                -17 * u[gv(cut=(0, 4))]
                + 308 * u[gv(cut=(1, 3))]
                + 5178 * u[gv(cut=(2, 2))]
                + 308 * u[gv(cut=(3, 1))]
                + -17 * u[gv(cut=(4, 0))]
            ) / 5760
        case 5:
            out = (
                0 * u[gv(cut=(0, 6))]
                + -17 * u[gv(cut=(1, 5))]
                + 308 * u[gv(cut=(2, 4))]
                + 5178 * u[gv(cut=(3, 3))]
                + 308 * u[gv(cut=(4, 2))]
                + -17 * u[gv(cut=(5, 1))]
                + 0 * u[gv(cut=(6, 0))]
            ) / 5760
        case 6:
            out = (
                (367 / 967680) * u[gv(cut=(0, 6))]
                + (-281 / 53760) * u[gv(cut=(1, 5))]
                + (6361 / 107520) * u[gv(cut=(2, 4))]
                + (215641 / 241920) * u[gv(cut=(3, 3))]
                + (6361 / 107520) * u[gv(cut=(4, 2))]
                + (-281 / 53760) * u[gv(cut=(5, 1))]
                + (367 / 967680) * u[gv(cut=(6, 0))]
            )
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
        case 8:
            out = (
                (-27859 / 464486400) * u[gv(cut=(0, 8))]
                + (49879 / 58060800) * u[gv(cut=(1, 7))]
                + (-801973 / 116121600) * u[gv(cut=(2, 6))]
                + (3629953 / 58060800) * u[gv(cut=(3, 5))]
                + (41208059 / 46448640) * u[gv(cut=(4, 4))]
                + (3629953 / 58060800) * u[gv(cut=(5, 3))]
                + (-801973 / 116121600) * u[gv(cut=(6, 2))]
                + (49879 / 58060800) * u[gv(cut=(7, 1))]
                + (-27859 / 464486400) * u[gv(cut=(8, 0))]
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


def second_order_central_difference(
    u: np.ndarray, axis: int, h: float = None
) -> np.ndarray:
    """
    compute second order central difference of u along axis
    args:
        u (array_like) : array of arbitrary shape
        axis (int) : along which to differentiate
        h (float) : grid spacing. if None, no scaling is applied
    returns:
        out (array_like) : second order central difference of u. shorter than u by 2 along specified axis
    """
    gv = partial(get_view, ndim=u.ndim, axis=axis)
    out = 0.5 * (u[gv(cut=(2, 0))] - u[gv(cut=(0, 2))])
    if h is not None:
        out /= h
    return out
