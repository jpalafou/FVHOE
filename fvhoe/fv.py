from itertools import product
from functools import lru_cache
from fvhoe.hydro import HydroState
from fvhoe.stencils import (
    get_conservative_interpolation_stencil_weights,
    get_transverse_reconstruction_stencil_weights,
)
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


@lru_cache(maxsize=None)
def get_symmetric_slices(ndim: int, nslices: int, axis: int) -> list:
    """
    generate a list of symmetric slices
    args:
        ndim (int) : number of dimensions of the sliced array
        nslices (int) : number of slices
        axis (int) : axis along which to slice
    returns:
        out (list) : list of slices from "left to right" along axis
    """
    if nslices < 1:
        raise ValueError(f"{nslices=}")
    hs = HydroState(ndim=ndim)
    out = [hs(axis=axis, cut=(i, -(nslices - 1) + i)) for i in range(nslices)]
    return out


def stencil_sweep(
    u: np.ndarray, mode: str, p: int, axis: int, pos: str = None
) -> np.ndarray:
    """
    args:
        u (array_like) : array of finite volume cell averages of arbitrary shape
        mode (str) : stencil mode
            "interpolation" : conservative interpolation
            "reconstruction" : transverse reconstruction
        p (int) : polynomial degree of conservative interpolation
        axis (int) : along which to interpolate
        pos (str) : interpolation position along finite volume
            "l" left
            "c" center
            "r" right
    returns:
        out (array_like) : array of interpolations/reconstructions
    """
    if mode == "interpolation":
        weights = get_conservative_interpolation_stencil_weights(p=p, pos=pos)
    elif mode == "reconstruction":
        weights = get_transverse_reconstruction_stencil_weights(p=p)
    else:
        raise ValueError(f"{mode=}")

    # get slices
    slices = get_symmetric_slices(u.ndim, len(weights), axis)

    # Pre-allocate output array with the same shape as u
    out = np.zeros_like(u[slices[0]])

    # Efficient summation loop
    for w, s in zip(weights, slices):
        np.add(out, w * u[s], out=out)  # Accumulate in place to save memory
    return out


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
    out = stencil_sweep(u=u, mode="interpolation", p=p, axis=axis, pos=pos)
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
    out = stencil_sweep(u=u, mode="reconstruction", p=p, axis=axis)
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
    slices = get_symmetric_slices(u.ndim, 3, axis)
    out = 0.5 * (u[slices[2]] - u[slices[0]])
    if h is not None:
        out /= h
    return out
