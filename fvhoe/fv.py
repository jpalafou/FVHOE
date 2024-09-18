from itertools import product
from functools import lru_cache
from fvhoe.array_manager import get_array_slice as slc
from fvhoe.stencils import (
    get_fv_conservative_weights,
    get_transverse_reconstruction_weights,
)
from numba import njit
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


@njit
def get_stencil_size(p: int, mode: str = "right") -> int:
    """
    Get the size of a stencil for a given polynomial degree.
    Args:
        p (int): Polynomial degree
        mode (str): Stencil mode
            "total": Total number of points in the stencil
            "right": Number of points to the right of the center
    Returns:
        int: Size of the stencil
    """
    if mode == "total":
        return -2 * (-p // 2) + 1
    elif mode == "right":
        return -(-p // 2)
    else:
        raise ValueError("Invalid mode. Use 'total' or 'right'.")


@njit
def get_numba_slc(arr: np.ndarray, axis: int, lidx: int, ridx: int) -> np.ndarray:
    """
    Get a view from the array along a specific axis.
    Args:
        arr (np.ndarray): The input array
        axis (int): Axis along which to slice
        lidx (int): Start index of the view
        ridx (int): End index of the view
    Returns:
        np.ndarray: Sliced view of the array
    """
    if lidx == 0 and ridx == 0:
        return arr
    if axis == 1:
        if lidx == 0 and ridx != 0:
            return arr[:, :ridx, :, :]
        elif lidx != 0 and ridx == 0:
            return arr[:, lidx:, :, :]
        else:
            return arr[:, lidx:ridx, :, :]
    elif axis == 2:
        if lidx == 0 and ridx != 0:
            return arr[:, :, :ridx, :]
        elif lidx != 0 and ridx == 0:
            return arr[:, :, lidx:, :]
        else:
            return arr[:, :, lidx:ridx, :]
    elif axis == 3:
        if lidx == 0 and ridx != 0:
            return arr[:, :, :, :ridx]
        elif lidx != 0 and ridx == 0:
            return arr[:, :, :, lidx:]
        else:
            return arr[:, :, :, lidx:ridx]
    else:
        raise ValueError("Invalid axis. Axis must be 1, 2, or 3.")


@njit
def fv_stencil_sweep(
    u: np.ndarray, p: int, axis: int, mode: str, pos: str = "l"
) -> np.ndarray:
    """
    Perform stencil sweep on a given array along a specific axis.
    Args:
        u (np.ndarray): Array of finite volume cell averages of arbitrary shape
        p (int): Polynomial degree of conservative interpolation
        axis (int): Axis along which to interpolate
        mode (str) : "interpolation" or "reconstruction"
        pos (str): Interpolation position along finite volume
            "l": Left
            "c": Center
            "r": Right
    Returns:
        np.ndarray: Array of interpolations
    """
    if axis not in {1, 2, 3}:
        raise ValueError("Invalid axis. axis must be 1, 2, or 3.")

    # get weights and number of slices
    if mode == "interpolation":
        weights = get_fv_conservative_weights(p, pos)
    elif mode == "reconstruction":
        weights = get_transverse_reconstruction_weights(p)
    nslices = get_stencil_size(p, mode="total")

    # Create an output array with the same shape as the sliced view
    slice_data0 = get_numba_slc(u, axis, 0, -(nslices - 1))
    out = np.zeros(slice_data0.shape, dtype=np.float64)

    # Perform the interpolation
    for w, i in zip(weights, range(nslices)):
        slice_data = get_numba_slc(u, axis, i, -(nslices - 1) + i)
        out += w * slice_data

    return out


def conservative_interpolation(
    u: np.ndarray, p: int, axis: int, pos: str
) -> np.ndarray:
    """
    perform conservative interpolation of u along axis
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
    return fv_stencil_sweep(u=u, p=p, axis=axis, mode="interpolation", pos=pos)


def transverse_reconstruction(u: np.ndarray, p: int, axis: int) -> np.ndarray:
    """
    perform transverse reconstruction of u along axis
    args:
        u (array_like) : array of pointwise interpolations of arbitrary shape
        p (int) : polynomial degree of integral interpolation
        axis (int) : along which to interpolate
    returns:
        out (array_like) : array of reconstructed face integrals
    """
    return fv_stencil_sweep(u=u, p=p, axis=axis, mode="reconstruction")


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
