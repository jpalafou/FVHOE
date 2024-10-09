from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from typing import Tuple, Union


@dataclass
class HydroState:
    """
    HydroState class for managing hydrodynamic variables and passive scalars.
        rho: density
        vx, vy, vz: velocity components
        mx, my, mz: momentum components
        P: pressure
        E: total energy
        your_passive_scalar1, your_passive_scalar2, ...: passive scalars
    args:
        passive_scalars (tuple) : tuple of passive scalar names
        ndim (int) : number of dimensions

    There are two hash issues with this class:
        1. This class uses a dummy hash method that always returns the same value.
        2. This class is mutable, but it is not intended to be mutable. If the class is
              modified after instantiation, the hash will not change, leading to
              potentially unexpected behavior in the __call__ since it uses lru_cache.
    """

    passive_scalars: tuple = ()
    ndim: int = 4

    def __post_init__(self):
        self.variable_map = {
            "rho": 0,
            "vx": 1,
            "mx": 1,
            "vy": 2,
            "my": 2,
            "vz": 3,
            "mz": 3,
            "v": np.arange(1, 4),
            "m": np.arange(1, 4),
            "P": 4,
            "E": 4,
        }

        if self.passive_scalars:
            for i, scalar in enumerate(self.passive_scalars, start=5):
                self.variable_map[scalar] = i
            self.variable_map["passive_scalars"] = np.arange(
                5, 5 + len(self.passive_scalars)
            )
            self.includes_passive_scalars = True
        else:
            self.includes_passive_scalars = False

    def __hash__(self):
        return hash("HydroState dummy hash")

    @lru_cache(maxsize=None)
    def __call__(
        self,
        var: Union[str, Tuple[str]] = None,
        x: Tuple[int, int] = None,
        y: Tuple[int, int] = None,
        z: Tuple[int, int] = None,
        axis: int = None,
        cut: Tuple[int, int] = None,
        step: int = None,
    ) -> Union[Tuple[slice], slice]:
        """
        Get the slice for the given variable and coordinates.
        args:
            var (str) : variable name or tuple of variable names. if None, all variables are selected
            x (Tuple[int, int]) : x-coordinate slice. if None, all x-coordinates are selected
            y (Tuple[int, int]) : y-coordinate slice. if None, all y-coordinates are selected
            z (Tuple[int, int]) : z-coordinate slice. if None, all z-coordinates are selected
            axis (int) : axis to cut, alternative to x, y, z
            cut (Tuple[int, int]) : slice along dimension specified by axis. ignored if axis is None
            step (int) : step size for the slice. ignored if axis is None
        returns:
            Tuple[slice] : slices for the given variable and coordinates with length equal to ndim.
                if ndim is 1, a single slice is returned
        """
        slices = [slice(None)] * self.ndim

        if var is not None:
            if isinstance(var, str):
                # retrieve single variable index
                if var not in self.variable_map:
                    raise ValueError(f"Variable '{var}' not found.")
                slices[0] = self.variable_map[var]
            elif isinstance(var, tuple):
                # retrieve multiple variable indices
                missing_vars = set(var) - set(self.variable_map.keys())
                if missing_vars:
                    raise ValueError(f"Variables not found: {missing_vars}")
                slices[0] = np.array(list(map(self.variable_map.get, var)))
            else:
                raise ValueError(f"Invalid type for var: {type(var)}")

        axes = [1, 2, 3, axis]
        axis_slices = [x, y, z, cut]
        for i, axis_slice in zip(axes, axis_slices):
            if axis_slice is not None:
                if i >= self.ndim:
                    raise ValueError(
                        f"Invalid axis {i} for array with {self.ndim} dimensions."
                    )
                if not isinstance(axis_slice, tuple):
                    raise ValueError(
                        f"Expected a tuple (start, stop) for axis {i}, got {axis_slice} of type {type(axis_slice)}"
                    )
                if len(axis_slice) != 2:
                    raise ValueError(
                        f"Invalid tuple length for axis {i}: {len(axis_slice)}"
                    )
                slices[i] = slice(
                    axis_slice[0] or None,
                    axis_slice[1] or None,
                    step if i == axis else None,
                )

        if len(slices) == 1:
            return slices[0]
        return tuple(slices)


# define 1D slicer with no passive scalars
hs = HydroState(ndim=1)


def compute_primitives(hs: HydroState, u: np.ndarray, gamma: float) -> np.ndarray:
    """
    args:
        hs (HydroState) : HydroState object
        u (array_like) : has variables names ["rho", "mx", "my", "mz", "E"]
        gamma (float) : specific heat ratio
    returns:
        w (array_like) : has variables names ["rho", "vx", "vy", "vz", "P"]
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

    if hs.includes_passive_scalars:
        w[hs("passive_scalars")] = u[hs("passive_scalars")] / rho

    return w


def compute_conservatives(hs: HydroState, w: np.ndarray, gamma: float) -> np.ndarray:
    """
    args:
        hs (HydroState) : HydroState object
        w (array_like) : has variables names ["rho", "vx", "vy", "vz", "P"]
        gamma (float) : specific heat ratio
    returns:
        u (array_like) : has variables names ["rho", "mx", "my", "mz", "E"]
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

    if hs.includes_passive_scalars:
        u[hs("passive_scalars")] = rho * w[hs("passive_scalars")]
    return u


def compute_sound_speed(w: np.ndarray, gamma: float, csq_floor: float) -> np.ndarray:
    """
    args:
        w (array_like) : has variables names ["rho", "P"]
        gamma (float) : specific heat ratio
        csq_floor (float) : floor on square of returned sound speed
    returns:
        out (array_like) : sound speeds
    """
    csq = gamma * w[hs("P")] / w[hs("rho")]
    out = np.sqrt(np.where(csq > csq_floor, csq, csq_floor))
    return out


def compute_fluxes(
    hs: HydroState,
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
        hs (HydroState) : HydroState object
        u (array_like) : has variables names ["rho", "mx", "my", "mz", "E"]
        w (array_like) : has variables names ["rho", "vx", "vy", "vz", "P"]
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", "z"
        include_pressure (bool) : whether to include pressure
    returns:
        out (array_like) : fluxes in specified direction, has variables names ["rho", "mx", "my", "mz", "E"]
    """
    out = np.empty_like(u)
    v = w[hs("v" + dim)]  # velocity in dim-direction

    # assign values
    out[hs("rho")] = v * w[hs("rho")]
    out[hs("m")] = v * u[hs("m")]
    out[hs("E")] = v * u[hs("E")]
    if include_pressure:
        out[hs(f"m{dim}")] = out[hs(f"m{dim}")] + w[hs("P")]  # pressure term
        out[hs("E")] = out[hs("E")] + v * w[hs("P")]  # pressure term
    if hs.includes_passive_scalars:
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
    va = np.abs(w[hs("v")])
    out = CFL * h / np.max(np.sum(va, axis=0) + ndim * c)
    out = out.item()  # in case of cupy
    return out
