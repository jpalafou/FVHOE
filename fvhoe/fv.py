from itertools import product
from functools import lru_cache
from fvhoe.array_manager import get_array_slice as slc
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


@lru_cache(maxsize=100)
def get_symmetric_slices(nslices: int, ndim: int, axis: int) -> list:
    """
    generate a list of symmetric slices
    args:
        nslices (int) : number of slices
        ndim (int) : number of dimensions of the sliced array
        axis (int) : axis along which to slice
    returns:
        out (list) : list of slices from "left to right" along axis
    """
    if nslices < 1:
        raise ValueError(f"{nslices=}")
    return [
        slc(ndim=ndim, axis=axis, cut=(i, -(nslices - 1) + i)) for i in range(nslices)
    ]


@lru_cache(maxsize=100)
def get_stencil_size(p: int, mode: str = "right") -> int:
    """
    get the size of a stencil for a given polynomial degree
    args:
        p (int) : polynomial degree
        mode (str) : stencil mode
            "total" : total number of points in the stencil
            "right" : number of points to the right of the center
    returns:
        int : size of the stencil
    """
    if mode == "total":
        return -2 * (-p // 2) + 1
    elif mode == "right":
        return -(-p // 2)
    else:
        raise ValueError(f"{mode=}")


def conservative_interpolation(
    u: np.ndarray, p: int, axis: int, pos: str = "c"
) -> np.ndarray:
    """
    args:
        u (array_like) : array of finite volume cell averages of arbitrary shape
        p (int) : polynomial degree of conservative interpolation
        axis (int) : along which to interpolate
        pos (str) : interpolation position along finite volume
            "l" left
            "c" center
            "r" right
    returns:
        out (array_like) : array of interpolations
    """
    slices = get_symmetric_slices(get_stencil_size(p, mode="total"), u.ndim, axis)

    if pos == "r":
        return conservative_interpolation(
            u=u[slc(ndim=u.ndim, axis=axis, cut=(0, 0), step=-1)],
            p=p,
            axis=axis,
            pos="l",
        )[slc(ndim=u.ndim, axis=axis, cut=(0, 0), step=-1)]

    match p:
        case 0:
            out = u.copy()
        case 1:
            if pos == "l":
                out = (1 * u[slices[0]] + 4 * u[slices[1]] + -1 * u[slices[2]]) / 4
            elif pos == "c":
                out = (0 * u[slices[0]] + 1 * u[slices[1]] + 0 * u[slices[2]]) / 1
        case 2:
            if pos == "l":
                out = (2 * u[slices[0]] + 5 * u[slices[1]] + -1 * u[slices[2]]) / 6
            elif pos == "c":
                out = (-1 * u[slices[0]] + 26 * u[slices[1]] + -1 * u[slices[2]]) / 24
        case 3:
            if pos == "l":
                out = (
                    -1 * u[slices[0]]
                    + 10 * u[slices[1]]
                    + 20 * u[slices[2]]
                    + -6 * u[slices[3]]
                    + 1 * u[slices[4]]
                ) / 24
            elif pos == "c":
                out = (
                    0 * u[slices[0]]
                    + -1 * u[slices[1]]
                    + 26 * u[slices[2]]
                    + -1 * u[slices[3]]
                    + 0 * u[slices[4]]
                ) / 24
        case 4:
            if pos == "l":
                out = (
                    -3 * u[slices[0]]
                    + 27 * u[slices[1]]
                    + 47 * u[slices[2]]
                    + -13 * u[slices[3]]
                    + 2 * u[slices[4]]
                ) / 60
            elif pos == "c":
                out = (
                    9 * u[slices[0]]
                    + -116 * u[slices[1]]
                    + 2134 * u[slices[2]]
                    + -116 * u[slices[3]]
                    + 9 * u[slices[4]]
                ) / 1920
        case 5:
            if pos == "l":
                out = (
                    (1 / 120) * u[slices[0]]
                    + (-1 / 12) * u[slices[1]]
                    + (59 / 120) * u[slices[2]]
                    + (47 / 60) * u[slices[3]]
                    + (-31 / 120) * u[slices[4]]
                    + (1 / 15) * u[slices[5]]
                    + (-1 / 120) * u[slices[6]]
                )
            elif pos == "c":
                out = (
                    0 * u[slices[0]]
                    + (3 / 640) * u[slices[1]]
                    + (-29 / 480) * u[slices[2]]
                    + (1067 / 960) * u[slices[3]]
                    + (-29 / 480) * u[slices[4]]
                    + (3 / 640) * u[slices[5]]
                    + 0 * u[slices[6]]
                )
        case 6:
            if pos == "l":
                out = (
                    (1 / 105) * u[slices[0]]
                    + (-19 / 210) * u[slices[1]]
                    + (107 / 210) * u[slices[2]]
                    + (319 / 420) * u[slices[3]]
                    + (-101 / 420) * u[slices[4]]
                    + (5 / 84) * u[slices[5]]
                    + (-1 / 140) * u[slices[6]]
                )
            elif pos == "c":
                out = (
                    (-5 / 7168) * u[slices[0]]
                    + (159 / 17920) * u[slices[1]]
                    + (-7621 / 107520) * u[slices[2]]
                    + (30251 / 26880) * u[slices[3]]
                    + (-7621 / 107520) * u[slices[4]]
                    + (159 / 17920) * u[slices[5]]
                    + (-5 / 7168) * u[slices[6]]
                )
        case 7:
            if pos == "l":
                out = (
                    (-1 / 560) * u[slices[0]]
                    + (17 / 840) * u[slices[1]]
                    + (-97 / 840) * u[slices[2]]
                    + (449 / 840) * u[slices[3]]
                    + (319 / 420) * u[slices[4]]
                    + (-223 / 840) * u[slices[5]]
                    + (71 / 840) * u[slices[6]]
                    + (-1 / 56) * u[slices[7]]
                    + (1 / 560) * u[slices[8]]
                )
            elif pos == "c":
                out = (
                    0 * u[slices[0]]
                    + (-5 / 7168) * u[slices[1]]
                    + (159 / 17920) * u[slices[2]]
                    + (-7621 / 107520) * u[slices[3]]
                    + (30251 / 26880) * u[slices[4]]
                    + (-7621 / 107520) * u[slices[5]]
                    + (159 / 17920) * u[slices[6]]
                    + (-5 / 7168) * u[slices[7]]
                    + 0 * u[slices[8]]
                )
        case 8:
            if pos == "l":
                out = (
                    (-1 / 504) * u[slices[0]]
                    + (11 / 504) * u[slices[1]]
                    + (-61 / 504) * u[slices[2]]
                    + (275 / 504) * u[slices[3]]
                    + (1879 / 2520) * u[slices[4]]
                    + (-641 / 2520) * u[slices[5]]
                    + (199 / 2520) * u[slices[6]]
                    + (-41 / 2520) * u[slices[7]]
                    + (1 / 630) * u[slices[8]]
                )
            elif pos == "c":
                out = (
                    (35 / 294912) * u[slices[0]]
                    + (-425 / 258048) * u[slices[1]]
                    + (31471 / 2580480) * u[slices[2]]
                    + (-100027 / 1290240) * u[slices[3]]
                    + (5851067 / 5160960) * u[slices[4]]
                    + (-100027 / 1290240) * u[slices[5]]
                    + (31471 / 2580480) * u[slices[6]]
                    + (-425 / 258048) * u[slices[7]]
                    + (35 / 294912) * u[slices[8]]
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

    slices = get_symmetric_slices(get_stencil_size(p, mode="total"), u.ndim, axis)

    match p:
        case 0:
            out = u.copy()
        case 1:
            out = (0 * u[slices[0]] + 1 * u[slices[1]] + 0 * u[slices[2]]) / 1
        case 2:
            out = (1 * u[slices[0]] + 22 * u[slices[1]] + 1 * u[slices[2]]) / 24
        case 3:
            out = (
                0 * u[slices[0]]
                + 1 * u[slices[1]]
                + 22 * u[slices[2]]
                + 1 * u[slices[3]]
                + 0 * u[slices[4]]
            ) / 24
        case 4:
            out = (
                -17 * u[slices[0]]
                + 308 * u[slices[1]]
                + 5178 * u[slices[2]]
                + 308 * u[slices[3]]
                + -17 * u[slices[4]]
            ) / 5760
        case 5:
            out = (
                0 * u[slices[0]]
                + -17 * u[slices[1]]
                + 308 * u[slices[2]]
                + 5178 * u[slices[3]]
                + 308 * u[slices[4]]
                + -17 * u[slices[5]]
                + 0 * u[slices[6]]
            ) / 5760
        case 6:
            out = (
                (367 / 967680) * u[slices[0]]
                + (-281 / 53760) * u[slices[1]]
                + (6361 / 107520) * u[slices[2]]
                + (215641 / 241920) * u[slices[3]]
                + (6361 / 107520) * u[slices[4]]
                + (-281 / 53760) * u[slices[5]]
                + (367 / 967680) * u[slices[6]]
            )
        case 7:
            out = (
                0 * u[slices[0]]
                + (367 / 967680) * u[slices[1]]
                + (-281 / 53760) * u[slices[2]]
                + (6361 / 107520) * u[slices[3]]
                + (215641 / 241920) * u[slices[4]]
                + (6361 / 107520) * u[slices[5]]
                + (-281 / 53760) * u[slices[6]]
                + (367 / 967680) * u[slices[7]]
                + 0 * u[slices[8]]
            )
        case 8:
            out = (
                (-27859 / 464486400) * u[slices[0]]
                + (49879 / 58060800) * u[slices[1]]
                + (-801973 / 116121600) * u[slices[2]]
                + (3629953 / 58060800) * u[slices[3]]
                + (41208059 / 46448640) * u[slices[4]]
                + (3629953 / 58060800) * u[slices[5]]
                + (-801973 / 116121600) * u[slices[6]]
                + (49879 / 58060800) * u[slices[7]]
                + (-27859 / 464486400) * u[slices[8]]
            )
        case _:
            raise NotImplementedError(f"{p=}")

    return out


def interpolate_cell_centers(
    u: np.ndarray, p: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    compute cell centers from an array of finite volume averages
    args:
        u (array_like) : array of finite volume averages, with shape (# vars, nx, ny, nz)
        p (Tuple[int, int, int]) : polynomial degrees in each direction (px, py, pz)
    returns:
        out (array_like) : interpolations of cell centers
    """
    out = conservative_interpolation(
        u=conservative_interpolation(
            u=conservative_interpolation(u=u, p=p[0], axis=1, pos="c"),
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


@lru_cache(maxsize=10)
def get_legendre_quadrature(p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    get Legendre quadrature points and weights, scaled to [0, 1]
    args:
        p (int) : polynomial degree
    returns:
        points (array_like) : quadrature points
        weights (array_like) : quadrature weights
    """
    points, weights = np.polynomial.legendre.leggauss(-(-(p + 1) // 2))
    points /= 2
    weights /= 2
    return points, weights


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
    points_and_weights = [get_legendre_quadrature(p_i) for p_i in p]

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
    slices = get_symmetric_slices(3, u.ndim, axis)
    out = 0.5 * (u[slices[2]] - u[slices[0]])
    if h is not None:
        out /= h
    return out
