from fvhoe.named_array import NamedNumpyArray
import numpy as np


def compute_primitives(u: NamedNumpyArray, gamma: float) -> NamedNumpyArray:
    """
    args:
        u (NamedArray) : has variables names ["rho", "mx", "my", "mz", "E"]
        gamma (float) : specific heat ratio
    returns:
        w (NamedArray) : has variables names ["rho", "vx", "vy", "vz", "P"]
    """
    w = u.rename_variables({"mx": "vx", "my": "vy", "mz": "vz", "E": "P"})
    w.rho = u.rho
    w.vx = u.mx / u.rho
    w.vy = u.my / u.rho
    w.vz = u.mz / u.rho
    w.P = (gamma - 1) * (u.E - 0.5 * (u.mx * w.vx + u.my * w.vy + u.mz * w.vz))
    return w


def compute_conservatives(w: NamedNumpyArray, gamma: float) -> NamedNumpyArray:
    """
    args:
        w (NamedArray) : has variables names ["rho", "vx", "vy", "vz", "P"]
        gamma (float) : specific heat ratio
    returns:
        u (NamedArray) : has variables names ["rho", "mx", "my", "mz", "E"]
    """
    u = w.rename_variables({"vx": "mx", "vy": "my", "vz": "mz", "P": "E"})
    u.rho = w.rho
    u.mx = w.rho * w.vx
    u.my = w.rho * w.vy
    u.mz = w.rho * w.vz
    u.E = w.P / (gamma - 1) + 0.5 * (u.mx * w.vx + u.my * w.vy + u.mz * w.vz)
    return u


def compute_sound_speed(w: NamedNumpyArray, gamma: float) -> np.ndarray:
    """
    args:
        w (NamedArray) : has variables names ["rho", "P"]
        gamma (float) : specific heat ratio
    returns:
        out (array_like) : sound speeds
    """
    out = np.sqrt(gamma * w.P / w.rho)
    return out


def compute_fluxes(
    u: NamedNumpyArray,
    w: NamedNumpyArray,
    gamma: float,
    dim: str,
    include_pressure: bool = True,
) -> NamedNumpyArray:
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
    out = u.copy()
    v = getattr(w, "v" + dim)  # velocity in dim-direction
    out.rho = v * w.rho
    out.mx = v * u.mx
    out.my = v * u.my
    out.mz = v * u.mz
    out.E = v * u.E
    if include_pressure:
        mflux = getattr(out, "m" + dim)
        setattr(out, "m" + dim, mflux + w.P)
        Eflux = getattr(out, "E")
        setattr(out, "E", Eflux + v * w.P)
    return out


def advection_dt(
    hx: float,
    vx: float,
    CFL: float = 0.8,
    hy: float = 1.0,
    hz: float = 1.0,
    vy: float = 0.0,
    vz: float = 0.0,
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
    out = CFL / (np.abs(vx / hx) + np.abs(vy / hy) + np.abs(vz / hz))
    return out


def hydro_dt(w: NamedNumpyArray, h: float, CFL: float, gamma: float) -> float:
    """
    compute suitable time-step size for Euler equations
    args:
        w (NamedArray) : primitive variables
        h (float) : mesh spacing
        CFL (float) : Courant-Friedrichs-Lewy condition
        gamma (float) : specific heat ratio
    returns:
        out (float) : time-step size
    """
    c = compute_sound_speed(w, gamma)
    vxa = np.abs(w.vx)
    vya = np.abs(w.vy)
    vza = np.abs(w.vz)
    out = CFL * h / np.max(np.maximum(np.maximum(vxa, vya), vza) + c).item()
    if out < 0:
        raise BaseException("Negative dt encountered.")
    return out
