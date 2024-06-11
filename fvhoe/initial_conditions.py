from fvhoe.config import primitive_names
from fvhoe.named_array import NamedNumpyArray
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
) -> NamedNumpyArray:
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
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    out = NamedNumpyArray(np.asarray([np.empty_like(x)] * 5), primitive_names)

    # assign density
    wave_axis = np.zeros_like(x)
    for dim_name, dim_arr in zip(["x", "y", "z"], [x, y, z]):
        if dim_name in dims:
            wave_axis += dim_arr
    density_range = rho_min_max[1] - rho_min_max[0]
    out.rho = (
        0.5 * density_range * np.sin(2 * np.pi * wave_axis)
        + 0.5 * density_range
        + rho_min_max[0]
    )

    # assign other variables
    out.P[...] = P
    out.vx[...] = vx
    out.vy[...] = vy
    out.vz[...] = vz
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
) -> NamedNumpyArray:
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
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    out = NamedNumpyArray(np.asarray([np.empty_like(x)] * 5), primitive_names)

    # assign density
    inside_square = np.ones_like(x)
    for dim_name, dim_arr in zip(["x", "y", "z"], [x, y, z]):
        if dim_name in dims:
            inside_square_1d = np.logical_and(dim_arr > 0.25, dim_arr < 0.75)
            inside_square = np.logical_and(inside_square, inside_square_1d)
    out.rho = np.where(inside_square, rho_min_max[1], rho_min_max[0])

    # assign other variables
    out.P[...] = P
    out.vx[...] = vx
    out.vy[...] = vy
    out.vz[...] = vz
    return out


def slotted_disk(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray = None,
    rho_min_max: Tuple[float, float] = (1, 2),
    P: float = 1,
) -> NamedNumpyArray:
    """
    slotted disk revolving around (0.5, 0.5)
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        rho_min_max (Tuple[float, float]) : min density, max density
        P (float) : uniform pressure
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    out = NamedNumpyArray(np.asarray([np.empty_like(x)] * 5), primitive_names)

    # density and velocity
    xc, yc = x - 0.5, y - 0.5
    rsq = np.square(xc) + np.square(y - 0.75)
    inside_disk = np.logical_and(
        rsq < 0.15**2, np.logical_not(np.logical_and(np.abs(xc) < 0.025, y < 0.85))
    )
    out.rho = np.where(inside_disk, rho_min_max[1], rho_min_max[0])
    out.vx[...] = -yc
    out.vy[...] = xc
    out.vz[...] = 0

    # other variables
    out.P[...] = P
    return out


def shock_tube_1d(
    x: np.ndarray,
    y: np.ndarray = None,
    z: np.ndarray = None,
    dim: str = "x",
    shock_position: float = 0.5,
    rho_left_right: Tuple[float, float] = (1, 0.125),
    v_left_right: Tuple[float, float] = (0, 0),
    P_left_right: Tuple[float, float] = (1, 0.1),
) -> NamedNumpyArray:
    """
    1D shock tube initial conditions (default is Sod's problem)
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        dim (str) : "x", "y", "z"
        shock_position (float) : position of the shock
        rho_left_right (Tuple[float, float]) : density on left and right of shock
        v_left_right (Tuple[float, float]) : velocity on left and right of shock
        P_left_right (Tuple[float, float]) : pressure on left and right of shock
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    if dim not in ["x", "y", "z"]:
        raise ValueError("dim must be 'x', 'y', or 'z'")
    axis = {"x": x, "y": y, "z": z}[dim]
    out = NamedNumpyArray(np.asarray([np.empty_like(x)] * 5), primitive_names)
    out.rho = np.where(axis < shock_position, rho_left_right[0], rho_left_right[1])
    v = np.where(axis < shock_position, v_left_right[0], v_left_right[1])
    out.vx[...] = v if dim == "x" else 0
    out.vy[...] = v if dim == "y" else 0
    out.vz[...] = v if dim == "z" else 0
    out.P = np.where(axis < shock_position, P_left_right[0], P_left_right[1])
    return out


def shock_tube_2d(
    x: np.ndarray,
    y: np.ndarray = None,
    z: np.ndarray = None,
    dims: str = "xy",
    center: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    radius: float = 0.1,
    rho_in_out: Tuple[float, float] = (1, 1),
    P_in_out: Tuple[float, float] = (1, 0.2),
) -> NamedNumpyArray:
    """
    cross section of 2D shock tube initial conditions
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        dims (str) : contains "x", "y", and/or "z"
        center (Tuple[float, float, float]) : center of the blast
        radius (float) : radius of the blast
        rho_in_out (Tuple[float, float]) : density inside and outside the blast
        P_in_out (Tuple[float, float]) : pressure inside and outside the blast
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    if len(dims) != 2:
        raise ValueError("dims must contain exactly 2 dimensions")
    if dims[0] not in ["x", "y", "z"] or dims[1] not in ["x", "y", "z"]:
        raise ValueError("dims must be 'xy', 'xz', or 'yz'")
    out = NamedNumpyArray(np.asarray([np.empty_like(x)] * 5), primitive_names)
    xc, yc, zc = x - center[0], y - center[1], z - center[2]
    rsq = np.zeros_like(xc)
    rsq += np.square(xc) if "x" in dims else 0
    rsq += np.square(yc) if "y" in dims else 0
    rsq += np.square(zc) if "z" in dims else 0
    inside_blast = rsq < radius**2
    out.rho[...] = np.where(inside_blast, rho_in_out[0], rho_in_out[1])
    out.vx[...] = 0
    out.vy[...] = 0
    out.vz[...] = 0
    out.P = np.where(inside_blast, P_in_out[0], P_in_out[1])
    return out
