from fvhoe.named_array import NamedNumpyArray
import numpy as np


def compute_primitives(u: NamedNumpyArray, gamma: float) -> NamedNumpyArray:
    """
    args:
        u (NamedArray) : has variables names ["rho", "px", "py", "pz", "E"]
        gamma (float) : specific heat ratio
    returns:
        w (NamedArray) : has variables names ["rho", "vx", "vy", "vz", "P"]
    """
    w = u.rename_variables({"px": "vx", "py": "vy", "pz": "vz", "E": "P"})
    w.rho = u.rho
    w.vx = u.px / u.rho
    w.vy = u.py / u.rho
    w.vz = u.pz / u.rho
    w.P = (gamma - 1) * (u.E - 0.5 * (u.px * w.vx + u.py * w.vy + u.pz * w.vz))
    return w


def compute_conservatives(w: NamedNumpyArray, gamma: float) -> NamedNumpyArray:
    """
    args:
        w (NamedArray) : has variables names ["rho", "vx", "vy", "vz", "P"]
        gamma (float) : specific heat ratio
    returns:
        u (NamedArray) : has variables names ["rho", "px", "py", "pz", "E"]
    """
    u = w.rename_variables({"vx": "px", "vy": "py", "vz": "pz", "P": "E"})
    u.rho = w.rho
    u.px = w.rho * w.vx
    u.py = w.rho * w.vy
    u.pz = w.rho * w.vz
    u.E = w.P / (gamma - 1) + 0.5 * (u.px * w.vx + u.py * w.vy + u.pz * w.vz)
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
        u (array_like) : has variables names ["rho", "px", "py", "pz", "E"]
        w (array_like) : has variables names ["rho", "vx", "vy", "vz", "P"]
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", "z"
        include_pressure (bool) : whether to include pressure
    returns:
        out (array_like) : fluxes in specified direction, has variables names ["rho", "px", "py", "pz", "E"]
    """
    out = u.copy()
    v = getattr(w, "v" + dim)  # velocity in dim-direction
    out.rho = v * w.rho
    out.px = v * u.px
    out.py = v * u.py
    out.pz = v * u.pz
    out.E = v * u.E
    if include_pressure:
        pflux = getattr(out, "p" + dim)
        setattr(out, "p" + dim, pflux + w.P)
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
    c_x = np.abs(w.vx) + c
    c_y = np.abs(w.vy) + c
    c_z = np.abs(w.vz) + c
    out = CFL * h / np.max(c_x + c_y + c_z).item()
    if out < 0:
        raise BaseException("Negative dt encountered.")
    return out
