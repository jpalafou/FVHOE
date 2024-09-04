from fvhoe.array_manager import get_array_slice as slc
import numpy as np


def compute_primitives(u: np.ndarray, gamma: float) -> np.ndarray:
    """
    args:
        u (array_like) : has variables names ["rho", "mx", "my", "mz", "E"]
        gamma (float) : specific heat ratio
    returns:
        w (array_like) : has variables names ["rho", "vx", "vy", "vz", "P"]
    """
    w = np.empty_like(u)

    # get slices
    rho = u[slc("rho")]
    mx = u[slc("mx")]
    my = u[slc("my")]
    mz = u[slc("mz")]
    E = u[slc("E")]
    vx = w[slc("vx")]
    vy = w[slc("vy")]
    vz = w[slc("vz")]

    # assign values
    w[slc("rho")] = rho
    vx[...] = mx / rho
    vy[...] = my / rho
    vz[...] = mz / rho
    w[slc("P")] = (gamma - 1) * (E - 0.5 * (mx * vx + my * vy + mz * vz))
    return w


def compute_conservatives(w: np.ndarray, gamma: float) -> np.ndarray:
    """
    args:
        w (array_like) : has variables names ["rho", "vx", "vy", "vz", "P"]
        gamma (float) : specific heat ratio
    returns:
        u (array_like) : has variables names ["rho", "mx", "my", "mz", "E"]
    """
    u = np.empty_like(w)

    # get slices
    rho = w[slc("rho")]
    vx = w[slc("vx")]
    vy = w[slc("vy")]
    vz = w[slc("vz")]
    P = w[slc("P")]
    mx = u[slc("mx")]
    my = u[slc("my")]
    mz = u[slc("mz")]

    # assign values
    u[slc("rho")] = rho
    mx[...] = rho * vx
    my[...] = rho * vy
    mz[...] = rho * vz
    u[slc("E")] = P / (gamma - 1) + 0.5 * (mx * vx + my * vy + mz * vz)
    return u


def compute_sound_speed(
    w: np.ndarray, gamma: float, rho_P_floor: bool = False
) -> np.ndarray:
    """
    args:
        w (array_like) : has variables names ["rho", "P"]
        gamma (float) : specific heat ratio
        rho_P_floor (bool) : whether to floor rho and P to 1e-16
    returns:
        out (array_like) : sound speeds
    """
    P = np.maximum(w[slc("P")], 1e-16) if rho_P_floor else w[slc("P")]
    rho = np.maximum(w[slc("rho")], 1e-16) if rho_P_floor else w[slc("rho")]
    out = np.sqrt(gamma * P / rho)
    return out


def compute_fluxes(
    u: np.ndarray,
    w: np.ndarray,
    gamma: float,
    dim: str,
    include_pressure: bool = True,
) -> np.ndarray:
    """
    Riemann Solvers and Numerical Methods for Fluid Dynamics by Toro
    Page 3
    args:
        u (array_like) : has variables names ["rho", "mx", "my", "mz", "E"]]
        w (array_like) : has variables names ["rho", "vx", "vy", "vz", "P"]
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", "z"
        include_pressure (bool) : whether to include pressure
    returns:
        out (array_like) : fluxes in specified direction, has variables names ["rho", "mx", "my", "mz", "E"]
    """
    out = np.empty_like(u)
    v = w[slc("v" + dim)]  # velocity in dim-direction

    # assign values
    out[slc("rho")] = v * w[slc("rho")]
    out[slc("mx")] = v * u[slc("mx")]
    out[slc("my")] = v * u[slc("my")]
    out[slc("mz")] = v * u[slc("mz")]
    out[slc("E")] = v * u[slc("E")]
    if include_pressure:
        mflux = out[slc("m" + dim)]
        Eflux = out[slc("E")]
        mflux[...] = mflux + w[slc("P")]  # pressure term
        Eflux[...] = Eflux + v * w[slc("P")]  # pressure term
    return out


def advection_dt(
    h: float,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    CFL: float = 0.8,
) -> float:
    """
    get time-step size satisfying a CFL for an advection problem
    args:
        hx (float) : mesh spacing in x-direction
        vx (float) : maximum advection velocity in x-direction
        CFL (float) : Courant-Friedrichs-Lewy condition
        hy (float) : mesh spacing in y-direction
        vy (float) : maximum advection velocity in y-direction
        hz (float) : mesh spacing in z-direction
        vz (float) : maximum advection velocity in z-direction
    returns:
        out (float) : time-step size satisfying CFL
    """
    return CFL * h / (vx + vy + vz)


def hydro_dt(
    w: np.ndarray,
    h: float,
    ndim: float,
    CFL: float,
    gamma: float,
    rho_P_sound_speed_floor: bool = False,
) -> float:
    """
    compute suitable time-step size for Euler equations
    args:
        w (array_like) : primitive variables
        h (float) : mesh spacing
        ndim (int) : number of dimensions
        CFL (float) : Courant-Friedrichs-Lewy condition
        gamma (float) : specific heat ratio
        rho_P_sound_speed_floor (bool) : whether to apply a floor to density and pressure when computing sound speed
    returns:
        out (float) : time-step size
    """
    c = compute_sound_speed(w, gamma, rho_P_floor=rho_P_sound_speed_floor)
    vxa = np.abs(w[slc("vx")])
    vya = np.abs(w[slc("vy")])
    vza = np.abs(w[slc("vz")])
    out = CFL * h / np.max(vxa + vya + vza + ndim * c).item()
    if out < 0:
        raise BaseException("Negative dt encountered.")
    elif out < 1e-16:
        raise BaseException("dt is less than 1e-16.")
    return out
