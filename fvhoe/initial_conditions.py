import numpy as np
from typing import Tuple


def sinus(
    x: np.ndarray,
    y: np.ndarray = None,
    z: np.ndarray = None,
    dims: str = "x",
    rho_min_max: Tuple[float, float] = (1, 2),
    P: float = 1,
    vx: float = 1,
    vy: float = 0,
    vz: float = 0,
):
    """
    smooth density sine wave initial condition for advection
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        dims (str) : contains "x", "y", and/or "z"
        rho_min_max (Tuple[float, float]) : min density, max density
        P (float) : uniform pressure
        vx (float) : uniform x-velocity
        vy (float) : uniform y-velocity
        vz (float) : uniform z-velocity
    returns:
        out (array_like) : primitive variable initial condition, shape (5, nx, ny, nz)
    """
    out = np.asarray([np.empty_like(x)] * 5)
    wave_axis = np.zeros_like(x)
    for dim_name, dim_arr in zip(["x", "y", "z"], [x, y, z]):
        if dim_name in dims:
            wave_axis += dim_arr
    density_range = rho_min_max[1] - rho_min_max[0]
    out[0] = (
        0.5 * density_range * np.sin(2 * np.pi * wave_axis)
        + 0.5 * density_range
        + rho_min_max[0]
    )
    out[1] = P
    out[2] = vx
    out[3] = vy
    out[4] = vz
    return out


def square(
    x: np.ndarray,
    y: np.ndarray = None,
    z: np.ndarray = None,
    dims: str = "x",
    rho_min_max: Tuple[float, float] = (1, 2),
    P: float = 1,
    vx: float = 1,
    vy: float = 0,
    vz: float = 0,
):
    """
    discontinuous density square initial condition for advection
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        dims (str) : contains "x", "y", and/or "z"
        rho_min_max (Tuple[float, float]) : min density, max density
        P (float) : uniform pressure
        vx (float) : uniform x-velocity
        vy (float) : uniform y-velocity
        vz (float) : uniform z-velocity
    returns:
        out (array_like) : primitive variable initial condition, shape (5, nx, ny, nz)
    """
    out = np.asarray([np.empty_like(x)] * 5)
    inside_square = np.ones_like(x)
    for dim_name, dim_arr in zip(["x", "y", "z"], [x, y, z]):
        if dim_name in dims:
            inside_square_1d = np.logical_and(dim_arr > 0.25, dim_arr < 0.75)
            inside_square = np.logical_and(inside_square, inside_square_1d)
    out[0] = np.where(inside_square, rho_min_max[1], rho_min_max[0])
    out[1] = P
    out[2] = vx
    out[3] = vy
    out[4] = vz
    return out


def slotted_disk(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray = None,
    rho_min_max: Tuple[float, float] = (1, 2),
    P: float = 1,
):
    """
    slotted disk revolving around (0.5, 0.5)
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        rho_min_max (Tuple[float, float]) : min density, max density
        P (float) : uniform pressure
    returns:
        out (array_like) : primitive variable initial condition, shape (5, nx, ny, nz)
    """
    xc, yc = x - 0.5, y - 0.75
    rsq = np.square(xc) + np.square(yc)
    inside_disk = np.logical_and(
        rsq < 0.15**2, np.logical_not(np.logical_and(np.abs(xc) < 0.025, y < 0.85))
    )
    out = np.asarray([np.empty_like(x)] * 5)
    out[0] = np.where(inside_disk, rho_min_max[1], rho_min_max[0])
    out[1] = P
    out[2] = -yc
    out[3] = xc
    out[4] = 0
    return out
