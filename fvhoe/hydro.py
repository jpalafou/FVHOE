import numpy as np


def compute_primitives(u: np.ndarray, gamma: float) -> np.ndarray:
    """
    args:
        u (array_like) : array of conservative variables
            density, energy, x-momentum, y-momentum, z-momentum
        gamma (float) : specific heat ratio
    returns:
        w (array_like) : array of primitive variables
            density, pressure, x-velocity, y-velocity, z-velocity
    """
    w = np.empty_like(u)
    w[0, ...] = u[0, ...]  # density
    w[2, ...] = u[2, ...] / u[0, ...]  # x-velocity
    w[3, ...] = u[3, ...] / u[0, ...]  # y-velocity
    w[4, ...] = u[4, ...] / u[0, ...]  # z-velocity
    w[1, ...] = (gamma - 1) * (
        u[1, ...]
        - 0.5 * (w[2, ...] * u[2, ...] + w[3, ...] * u[3, ...] + w[4, ...] * u[4, ...])
    )  # pressure
    return w


def compute_conservatives(w: np.ndarray, gamma: float) -> np.ndarray:
    """
    args:
        w (array_like) : array of primitive variables
            density, pressure, x-velocity, y-velocity, z-velocity
        gamma (float) : specific heat ratio
    returns:
        u (array_like) : array of conservative variables
            density, energy, x-momentum, y-momentum, z-momentum
    """
    u = np.empty_like(w)
    u[0, ...] = w[0, ...]  # density
    u[2, ...] = w[0, ...] * w[2, ...]  # x-momentum
    u[3, ...] = w[0, ...] * w[3, ...]  # y-momentum
    u[4, ...] = w[0, ...] * w[4, ...]  # z-momentum
    u[1, ...] = w[1, ...] / (gamma - 1) + 0.5 * (
        w[2, ...] * u[2, ...] + w[3, ...] * u[3, ...] + w[4, ...] * u[4, ...]
    )  # energy
    return u


def compute_sound_speed(w: np.ndarray, gamma: float) -> np.ndarray:
    """
    args:
        w (array_like) : array of primitive variables
            density, pressure, x-velocity, y-velocity, z-velocity
    returns:
        out (array_like) : sound speed
    """
    out = np.sqrt(gamma * w[1, ...] / w[0, ...])
    return out


def compute_fluxes(u: np.ndarray, w: np.ndarray, gamma: float, dir: str) -> np.ndarray:
    """
    Riemann Solvers and Numerical Methods for Fluid Dynamics by Toro
    Page 3
    args:
        u (array_like) : array of conservative variables
            density, energy, x-momentum, y-momentum, z-momentum
        w (array_like) : array of primitive variables
            density, pressure, x-velocity, y-velocity, z-velocity
        gamma (float) : specific heat ratio
        dir (str) : "x", "y", "z"
    returns:
        out (array_like) : fluxes in specified direction
    """
    if dir == "x":
        F = np.empty_like(w)
        F[0, ...] = u[2, ...]  # rho u
        F[2, ...] = w[2, ...] * u[2, ...] + w[1, ...]  # rho u^2 + p
        F[3, ...] = w[3, ...] * u[2, ...]  # rho u v
        F[4, ...] = w[4, ...] * u[2, ...]  # rho u w
        F[1, ...] = w[2, ...] * (u[1, ...] + w[1, ...])  # u (E + p)
        out = F
    elif dir == "y":
        G = np.empty_like(w)
        G[0, ...] = u[3, ...]  # rho v
        G[2, ...] = w[2, ...] * u[3, ...]  # rho u v
        G[3, ...] = w[3, ...] * u[3, ...] + w[1, ...]  # rho v^2 + p
        G[4, ...] = w[4, ...] * u[3, ...]  # rho v w
        G[1, ...] = w[3, ...] * (u[1, ...] + w[1, ...])  # v (E + p)
        out = G
    elif dir == "z":
        H = np.empty_like(w)
        H[0, ...] = u[4, ...]  # rho w
        H[2, ...] = w[2, ...] * u[4, ...]  # rho u w
        H[3, ...] = w[3, ...] * u[4, ...]  # rho v w
        H[4, ...] = w[4, ...] * u[4, ...] + w[1, ...]  # rho w^2 + p
        H[1, ...] = w[4, ...] * (u[1, ...] + w[1, ...])  # w (E + p)
        out = H
    return out
