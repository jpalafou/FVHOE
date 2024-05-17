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
    w = u.copy()
    w.rename_variables({"px": "vx", "py": "vy", "pz": "vz", "E": "P"})
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
    u = w.copy()
    u.rename_variables({"vx": "px", "vy": "py", "vz": "pz", "P": "E"})
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
    u: NamedNumpyArray, w: NamedNumpyArray, gamma: float, dim: str
) -> NamedNumpyArray:
    """
    Riemann Solvers and Numerical Methods for Fluid Dynamics by Toro
    Page 3
    args:
        u (array_like) : has variables names ["rho", "px", "py", "pz", "E"]
        w (array_like) : has variables names ["rho", "vx", "vy", "vz", "P"]
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", "z"
    returns:
        out (array_like) : fluxes in specified direction, has variables names ["rho", "px", "py", "pz", "E"]
    """
    out = u.copy()
    p = getattr(u, "p" + dim)  # momentum in dim-direction
    out.rho = p
    out.px = p * w.vx + (w.p if dim == "x" else 0.0)
    out.py = p * w.vy + (w.p if dim == "y" else 0.0)
    out.pz = p * w.vz + (w.p if dim == "z" else 0.0)
    out.E = p * (u.E + w.P)
    return out


def advection_dt(
    hx: float,
    vx: float,
    C: float = 0.8,
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
        C (float) : Courant-Friedrichs-Lewy condition
        hy (float) : mesh spacing in y-direction
        vy (float) : maximum advection velocity in y-direction
        hz (float) : mesh spacing in z-direction
        vz (float) : maximum advection velocity in z-direction
    returns:
        out (float) : time-step size satisfying CFL
    """
    out = C / (np.abs(vx / hx) + np.abs(vy / hy) + np.abs(vz / hz))
    return out
