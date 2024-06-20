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
    out.P = P
    out.vx = vx
    out.vy = vy
    out.vz = vz
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
    out.P = P
    out.vx = vx
    out.vy = vy
    out.vz = vz
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
    out.vx = -yc
    out.vy = xc
    out.vz = 0

    # other variables
    out.P = P
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
    out.vx = v if dim == "x" else 0
    out.vy = v if dim == "y" else 0
    out.vz = v if dim == "z" else 0
    out.P = np.where(axis < shock_position, P_left_right[0], P_left_right[1])
    return out


def double_shock_1d(
    x: np.ndarray,
    y: np.ndarray = None,
    z: np.ndarray = None,
    dim: str = "x",
    shock_positions: Tuple[float, float] = (0.1, 0.9),
    rhos: Tuple[float, float, float] = (1, 1, 1),
    vs: Tuple[float, float, float] = (0, 0, 0),
    Ps: Tuple[float, float, float] = (10**3, 10**-2, 10**2),
):
    if dim not in ["x", "y", "z"]:
        raise ValueError("dim must be 'x', 'y', or 'z'")
    axis = {"x": x, "y": y, "z": z}[dim]
    out = NamedNumpyArray(np.asarray([np.empty_like(x)] * 5), primitive_names)
    out.rho = np.where(
        axis < shock_positions[0],
        rhos[0],
        np.where(axis < shock_positions[1], rhos[1], rhos[2]),
    )
    v = np.where(
        axis < shock_positions[0],
        vs[0],
        np.where(axis < shock_positions[1], vs[1], vs[2]),
    )
    out.vx = v if dim == "x" else 0
    out.vy = v if dim == "y" else 0
    out.vz = v if dim == "z" else 0
    out.P = np.where(
        axis < shock_positions[0],
        Ps[0],
        np.where(axis < shock_positions[1], Ps[1], Ps[2]),
    )
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
    out.rho = np.where(inside_blast, rho_in_out[0], rho_in_out[1])
    out.vx = 0
    out.vy = 0
    out.vz = 0
    out.P = np.where(inside_blast, P_in_out[0], P_in_out[1])
    return out


def shu_osher_1d(
    x: np.ndarray, y: np.ndarray = None, z: np.ndarray = None, dims: str = "x"
) -> NamedNumpyArray:
    """
    Shu Osher initial condition for advection on domain [0, 10]
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        dims (str) : contains "x", "y", or "z"
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    if dims not in ["x", "y", "z"]:
        raise ValueError("dims must be 'x', 'y', or 'z'")
    xr = {"x": x, "y": y, "z": z}[dims] - 5

    out = NamedNumpyArray(np.asarray([np.empty_like(x)] * 5), primitive_names)
    out.rho = np.where(xr < -4, 3.857143, 1 + 0.2 * np.sin(5 * xr))

    v = np.where(xr < -4, 2.629369, 0)
    out.vx = v if dims == "x" else 0
    out.vy = v if dims == "y" else 0
    out.vz = v if dims == "z" else 0
    out.P = np.where(xr < -4, 10.33333, 1)

    return out


def kelvin_helmholtz_2d(
    x: np.ndarray,
    y: np.ndarray = None,
    z: np.ndarray = None,
    sigma: float = 0.05 * np.sqrt(2),
    w0: float = 0.1,
) -> NamedNumpyArray:
    """
    2D Kelvin-Helmholtz instability
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        sigma (float) : vy perturbation amplitude
        w0 (float) : vy perturbation exponential amplitude
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    inner_region = np.logical_and(0.25 < y, y < 0.75)

    out = NamedNumpyArray(np.asarray([np.empty_like(x)] * 5), primitive_names)
    out.rho = np.where(inner_region, 2, 1)
    out.vx = np.where(inner_region, 0.5, -0.5)
    out.vy = (
        w0
        * np.sin(4 * np.pi * x)
        * (
            np.exp(-np.square(y - 0.25) / (2 * sigma * sigma))
            + np.exp(-np.square(y - 0.75) / (2 * sigma * sigma))
        )
    )
    out.vz = 0
    out.P = 2.5

    return out


def double_mach_reflection_2d(
    x: np.ndarray, y: np.ndarray = None, z: np.ndarray = None
) -> np.ndarray:
    """
    Double Mach reflection
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    xc = 1 / 6
    theta = np.pi / 3
    xp = x - y / np.tan(theta)
    gamma = 1.4

    out = NamedNumpyArray(np.asarray([np.empty_like(x)] * 5), primitive_names)
    out.rho = np.where(xp < xc, 8, gamma)
    out.vx = np.where(xp < xc, 7.145, 0)
    out.vy = np.where(xp < xc, -8.25 / 2, 0)
    out.vz = 0
    out.P = np.where(xp < xc, 116.5, 1)

    return out
