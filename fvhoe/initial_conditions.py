from fvhoe.config import conservative_names, primitive_names
from fvhoe.named_array import NamedNumpyArray
import numpy as np
from typing import Tuple, Union


def isnumeric(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def empty_NamedArray(
    shape: Tuple[int, int, int] = None, vars: str = "primitive"
) -> NamedNumpyArray:
    """
    assign primitive variables to a 3D mesh from uniform values, arrays, or functions of x, y, z
    args:
        shape (Tuple[int, int, int]) : defines the shape of the output, not including the variable dimension
        vars (str) : "primitive" or "conservative"
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    names = {"conservative": conservative_names, "primitive": primitive_names}[vars]
    out = NamedNumpyArray(np.asarray([np.empty(shape)] * 5), names)
    return out


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
    smooth density sine wave initial condition for advection.
    density is a scale of a sine wave in x, y, and/or z.
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
    rho_min, rho_range = rho_min_max[0], rho_min_max[1] - rho_min_max[0]
    r = np.zeros_like(x)
    for dim, coord in zip("xyz", [x, y, z]):
        r += coord if dim in dims else 0
    rho = rho_range * (0.5 * np.sin(2 * np.pi * r) + 0.5) + rho_min
    out = empty_NamedArray(r.shape)
    out.rho = rho
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
    density is a cube in x, y, and/or z bounded by [0.25, 0.75].
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        dims (str) : contains "x", "y", and/or "z"
        rho_min_max (Tuple[float, float]) : min density (outside cube), max density (inside cube)
        P (float) : uniform pressure
        vx (float) : uniform x-velocity
        vy (float) : uniform y-velocity
        vz (float) : uniform z-velocity
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    inside_square = np.ones_like(x, dtype=bool)
    for dim, r in zip("xyz", [x, y, z]):
        inside_square &= np.logical_and(r > 0.25, r < 0.75) if dim in dims else True
    rho = np.where(inside_square, rho_min_max[1], rho_min_max[0])
    out = empty_NamedArray(inside_square.shape)
    out.rho = rho
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
    slotted disk revolving around (0.5, 0.5) in the x-y plane
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        rho_min_max (Tuple[float, float]) : min density (outside disk), max density (inside disk)
        P (float) : uniform pressure
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    xc, yc = x - 0.5, y - 0.5
    rsq = np.square(xc) + np.square(y - 0.75)
    inside_disk = rsq < 0.15**2
    inside_disk &= np.logical_not(np.logical_and(np.abs(xc) < 0.025, y < 0.85))
    out = empty_NamedArray(inside_disk.shape)
    out.rho = np.where(inside_disk, rho_min_max[1], rho_min_max[0])
    out.P = P
    out.vx = -yc
    out.vy = xc
    out.vz = 0
    return out


def shock_1d(
    x: np.ndarray,
    y: np.ndarray = None,
    z: np.ndarray = None,
    dim: str = "x",
    position: float = 0.5,
    rho_left_right: Tuple[float, float] = (1, 0.125),
    P_left_right: Tuple[float, float] = (1, 0.1),
    v_left_right: Tuple[float, float] = (0, 0),
) -> NamedNumpyArray:
    """
    normal shock tube initial condition
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        dim (str) : "x", "y", or "z"
        position (float) : position of the shock
        rho_left_right (Tuple[float, float]) : density left and right of the shock
        P_left_right (Tuple[float, float]) : pressure left and right of the shock
        v_left_right (Tuple[float, float]) : velocity left and right of the shock
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    if dim not in "xyz":
        raise ValueError("dim must be 'x', 'y', or 'z'")
    r = {"x": x, "y": y, "z": z}[dim]
    out = empty_NamedArray(r.shape)
    out.rho = np.where(r < position, rho_left_right[0], rho_left_right[1])
    v = np.where(r < position, v_left_right[0], v_left_right[1])
    out.vx = v if dim == "x" else 0
    out.vy = v if dim == "y" else 0
    out.vz = v if dim == "z" else 0
    out.P = np.where(r < position, P_left_right[0], P_left_right[1])
    return out


def double_shock_1d(
    x: np.ndarray,
    y: np.ndarray = None,
    z: np.ndarray = None,
    dim: str = "x",
    positions: Tuple[float, float] = (0.1, 0.9),
    rhos: Tuple[float, float, float] = (1, 1, 1),
    vs: Tuple[float, float, float] = (0, 0, 0),
    Ps: Tuple[float, float, float] = (10**3, 10**-2, 10**2),
) -> NamedNumpyArray:
    """
    double shock tube initial condition
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        dim (str) : "x", "y", or "z"
        positions (Tuple[float, float]) : positions of the two shocks
        rhos (Tuple[float, float, float]) : densities left, between, and right of the shocks
        vs (Tuple[float, float, float]) : velocities left, between, and right of the shocks
        Ps (Tuple[float, float, float]) : pressures left, between, and right of the shocks
    """
    r = {"x": x, "y": y, "z": z}[dim]

    def f(r, values):
        return np.where(
            r < positions[0],
            values[0],
            np.where(r < positions[1], values[1], values[2]),
        )

    out = empty_NamedArray(r.shape)
    out.rho = f(r, rhos)
    out.P = f(r, Ps)
    out.vx = f(r, vs) if dim == "x" else 0
    out.vy = f(r, vs) if dim == "y" else 0
    out.vz = f(r, vs) if dim == "z" else 0
    return out


def shu_osher_1d(
    x: np.ndarray, y: np.ndarray = None, z: np.ndarray = None, dim: str = "x"
) -> NamedNumpyArray:
    """
    Shu Osher initial condition for advection on domain [0, 10]
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        dim (str) : "x", "y", or "z"
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    r = {"x": x, "y": y, "z": z}[dim] - 5
    v = v = np.where(r < -4, 2.629369, 0)
    out = empty_NamedArray(r.shape)
    out.rho = np.where(r < -4, 3.857143, 1 + 0.2 * np.sin(5 * r))
    out.vx = v if dim == "x" else 0
    out.vy = v if dim == "y" else 0
    out.vz = v if dim == "z" else 0
    out.P = np.where(r < -4, 10.33333, 1)
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
    out = empty_NamedArray(x.shape)
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
    x: np.ndarray,
    y: np.ndarray = None,
    z: np.ndarray = None,
    gamma: float = 1.4,
) -> np.ndarray:
    """
    Double Mach reflection
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        gamma (float) : specific heat ratio
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    xc = 1 / 6
    theta = np.pi / 3
    xp = x - y / np.tan(theta)
    out = empty_NamedArray(x.shape)
    out.rho = np.where(xp < xc, 8, gamma)
    out.vx = np.where(xp < xc, 7.145, 0)
    out.vy = np.where(xp < xc, -8.25 / 2, 0)
    out.vz = 0
    out.P = np.where(xp < xc, 116.5, 1)

    return out


def shock_tube(
    x: np.ndarray,
    y: np.ndarray = None,
    z: np.ndarray = None,
    dims: str = "x",
    center: Union[float, Tuple[float]] = 0.0,
    radius: float = 0.5,
    rho_in_out: Tuple[float, float] = (1, 0.125),
    P_in_out: Tuple[float, float] = (1, 0.1),
    vx_in_out: Tuple[float, float] = (0, 0),
    vy_in_out: Tuple[float, float] = (0, 0),
    vz_in_out: Tuple[float, float] = (0, 0),
    mode: str = "tube",
    x_cube: Tuple[float, float] = None,
    y_cube: Tuple[float, float] = None,
    z_cube: Tuple[float, float] = None,
    conservative: bool = False,
) -> NamedNumpyArray:
    """
    multidimensional shock tube initial condition. can be a sphere or a rectangular prism in 3D
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        dims (str) : contains "x", "y", and/or "z"
        center (Union[float, Tuple[float]]) : center of the blast
        radius (float) : radius of the blast
        rho_in_out (Tuple[float, float]) : density inside and outside the blast
        P_in_out (Tuple[float, float]) : pressure inside and outside the blast. if conservative, this is energy
        vx_in_out (Tuple[float, float]) : x-velocity inside and outside the blast. if conservative, this is momentum
        vy_in_out (Tuple[float, float]) : y-velocity inside and outside the blast. if conservative, this is momentum
        vz_in_out (Tuple[float, float]) : z-velocity inside and outside the blast. if conservative, this is momentum
        mode (str) : 'tube' or 'cube'. if 'cube', the blast is a rectangular prism and dims, center, and radius are ignored
        x_cube (Tuple[float, float]) : x-cube dimensions for mode='cube'
        y_cube (Tuple[float, float]) : y-cube dimensions for mode='cube'
        z_cube (Tuple[float, float]) : z-cube dimensions for mode='cube'
        conservative (bool) : if True, uses primitive arguments to define their conservative counterparts
    """
    if mode == "tube":
        # convert single values to tuples
        center_as_tuple = (center,) if isnumeric(center) else center

        # define center coordinates in 3D
        if len(center_as_tuple) != len(dims):
            raise ValueError("center must have the same length as dims")
        center_3d_coords = [np.nan, np.nan, np.nan]
        for i, dim in enumerate(dims):
            center_3d_coords["xyz".index(dim)] = center_as_tuple[i]

        # define radius in 3D
        xc = x - center_3d_coords[0] if "x" in dims else 0
        yc = y - center_3d_coords[1] if "y" in dims else 0
        zc = z - center_3d_coords[2] if "z" in dims else 0
        rsq = xc**2 + yc**2 + zc**2
        inside_region = rsq <= radius**2

    elif mode == "cube":
        if x_cube is None and y_cube is None and z_cube is None:
            raise ValueError("at least one cube dimension must be specified")
        # define cube coordinates in 3D
        inside_region = np.ones_like(x, dtype=bool)
        if x_cube is not None:
            inside_region &= np.logical_and(x >= x_cube[0], x <= x_cube[1])
        if y_cube is not None:
            inside_region &= np.logical_and(y >= y_cube[0], y <= y_cube[1])
        if z_cube is not None:
            inside_region &= np.logical_and(z >= z_cube[0], z <= z_cube[1])

    # define initial conditions
    out = empty_NamedArray(
        x.shape,
        vars="conservative" if conservative else "primitive",
    )
    out.rho = np.where(inside_region, rho_in_out[0], rho_in_out[1])
    if conservative:
        out.E = np.where(inside_region, P_in_out[0], P_in_out[1])
        out.mx = np.where(inside_region, vx_in_out[0], vx_in_out[1])
        out.my = np.where(inside_region, vy_in_out[0], vy_in_out[1])
        out.mz = np.where(inside_region, vz_in_out[0], vz_in_out[1])
    else:
        out.E = np.where(inside_region, P_in_out[0], P_in_out[1])
        out.vx = np.where(inside_region, vx_in_out[0], vx_in_out[1])
        out.vy = np.where(inside_region, vy_in_out[0], vy_in_out[1])
        out.vz = np.where(inside_region, vz_in_out[0], vz_in_out[1])
    return out


def sedov(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mode: str = "corner",
    dims: str = "xy",
    rho0: float = 1.0,
    E0: float = 1e-5,
    E1: float = 1.0,
) -> NamedNumpyArray:
    """
    Sedov blast wave initial condition in conservative variable form
    run with conservative_ic=True and fv_ic=True
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
        mode (str) : 'corner' or 'center'. if 'center', the blast is centered at the domain center
        dims (str) : contains "x", "y", and/or "z"
        rho0 (float) : background pressure
        E0 (float) : background energy
        E1 (float) : peak energy
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    # get mesh info and peak mesh energy
    Nx, Ny, Nz = x.shape
    hx = np.mean(x[1:, :, :] - x[:-1, :, :])
    hy = np.mean(y[:, 1:, :] - y[:, :-1, :])
    hz = np.mean(z[:, :, 1:] - z[:, :, :-1])
    Emax = E1
    Emax /= 2 * hx if "x" in dims else 1
    Emax /= 2 * hy if "y" in dims else 1
    Emax /= 2 * hz if "z" in dims else 1

    # define initial conditions
    out = empty_NamedArray(x.shape, "conservative")
    out.rho = rho0
    out.mx = 0
    out.my = 0
    out.mz = 0
    out.E = E0

    # place peak energy based on mode
    if mode == "corner":
        peak = (0, 0, 0)
    elif mode == "center":
        peak = (
            slice(None) if "x" not in dims else slice(Nx // 2 - 1, Nx // 2 + 1),
            slice(None) if "y" not in dims else slice(Ny // 2 - 1, Ny // 2 + 1),
            slice(None) if "z" not in dims else slice(Nz // 2 - 1, Nz // 2 + 1),
        )
    out.E[peak] = Emax

    return out


def athena_blast(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> NamedNumpyArray:
    """
    Athena test blast https://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
    x in [-0.5, 0.5]
    y in [-0.75, 0.75]
    periodic boundary conditions
    args:
        x (array_like) : 3D mesh of x-points, shape (nx, ny, nz)
        y (array_like) : 3D mesh of y-points, shape (nx, ny, nz)
        z (array_like) : 3D mesh of z-points, shape (nx, ny, nz)
    returns:
        out (NamedNumpyArray) : has variable names ["rho", "vx", "vy", "vz", "P"]
    """
    # radius centered at (0, 0)
    r = np.sqrt(np.square(x) + np.square(y))

    # define initial conditions
    out = empty_NamedArray(x.shape)
    out.rho = 1
    out.vx = 0
    out.vy = 0
    out.vz = 0
    out.P = np.where(r < 0.1, 10, 0.1)

    return out
