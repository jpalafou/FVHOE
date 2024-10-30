from fvhoe.array_management import HydroState
import numpy as np

# define 1D slicer with no passive scalars
_hs = HydroState(ndim=1)


def compute_primitives(hs: HydroState, u: np.ndarray, gamma: float) -> np.ndarray:
    """
    args:
        hs (HydroState) : HydroState object
        u (array_like) : has variables names ["rho", "mx", "my", "mz", "E", ...]
        gamma (float) : specific heat ratio
    returns:
        w (array_like) : has variables names ["rho", "vx", "vy", "vz", "P", ...]
    """
    w = np.empty_like(u)

    # get slices
    rho = u[hs("rho")]
    m = u[hs("m")]
    E = u[hs("E")]

    # assign values
    w[hs("rho")] = rho
    w[hs("v")] = m / rho
    w[hs("P")] = (gamma - 1) * (E - 0.5 * np.sum(m * w[hs("v")], axis=0))

    if hs.includes_passives:
        w[hs("passive_scalars")] = u[hs("passive_scalars")] / rho

    return w


def compute_conservatives(hs: HydroState, w: np.ndarray, gamma: float) -> np.ndarray:
    """
    args:
        hs (HydroState) : HydroState object
        w (array_like) : has variables names ["rho", "vx", "vy", "vz", "P", ...]
        gamma (float) : specific heat ratio
    returns:
        u (array_like) : has variables names ["rho", "mx", "my", "mz", "E", ...]
    """
    u = np.empty_like(w)

    # get slices
    rho = w[hs("rho")]
    v = w[hs("v")]
    P = w[hs("P")]

    # assign values
    u[hs("rho")] = rho
    u[hs("m")] = rho * v
    u[hs("E")] = P / (gamma - 1) + 0.5 * np.sum(u[hs("m")] * v, axis=0)

    if hs.includes_passives:
        u[hs("passive_scalars")] = rho * w[hs("passive_scalars")]
    return u


def compute_sound_speed(w: np.ndarray, gamma: float, csq_floor: float) -> np.ndarray:
    """
    args:
        w (array_like) : has variables names ["rho", "P", ...]
        gamma (float) : specific heat ratio
        csq_floor (float) : floor on square of returned sound speed
    returns:
        out (array_like) : sound speeds
    """
    csq = gamma * w[_hs("P")] / w[_hs("rho")]
    out = np.sqrt(np.where(csq > csq_floor, csq, csq_floor))
    return out


def compute_fluxes(
    hs: HydroState,
    w: np.ndarray,
    u: np.ndarray,
    gamma: float,
    dim: str,
    include_pressure: bool = True,
) -> np.ndarray:
    """
    Riemann Solvers and Numerical Methods for Fluid Dynamics by Toro
    Page 3
    args:
        hs (HydroState) : HydroState object
        w (array_like) : array of primitive variables, has variables names ["rho", "vx", "vy", "vz", "P", ...]
        u (array_like) : array of conservative variables, has variables names ["rho", "mx", "my", "mz", "E", ...]
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", "z"
        include_pressure (bool) : whether to include pressure
    returns:
        out (array_like) : fluxes in specified direction, has variables names ["rho", "mx", "my", "mz", "E", ...]
    """
    out = np.empty_like(w)
    v = w[hs("v" + dim)]  # velocity in dim-direction

    # assign values
    out[hs("rho")] = v * w[hs("rho")]
    out[hs("m")] = v * u[hs("m")]
    out[hs("E")] = v * u[hs("E")]
    if include_pressure:
        out[hs(f"m{dim}")] = out[hs(f"m{dim}")] + w[hs("P")]  # pressure term
        out[hs("E")] = out[hs("E")] + v * w[hs("P")]  # pressure term
    if hs.includes_passives:
        out[hs("passive_scalars")] = v * u[hs("passive_scalars")]
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
        h (float) : mesh spacing
        vx (float) : maximum advection velocity in x-direction
        vy (float) : maximum advection velocity in y-direction
        vz (float) : maximum advection velocity in z-direction
        CFL (float) : Courant-Friedrichs-Lewy condition
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
    csq_floor: float,
) -> float:
    """
    compute suitable time-step size for Euler equations
    args:
        w (array_like) : primitive variables
        h (float) : mesh spacing
        ndim (int) : number of dimensions
        CFL (float) : Courant-Friedrichs-Lewy condition
        gamma (float) : specific heat ratio
        csq_floor (float) : floor on square of returned sound speed
    returns:
        out (float) : time-step size
    """
    c = compute_sound_speed(w, gamma, csq_floor=csq_floor)
    va = np.abs(w[_hs("v")])
    out = CFL * h / np.max(np.sum(va, axis=0) + ndim * c)
    out = out.item()  # in case of cupy
    return out
