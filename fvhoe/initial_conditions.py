from abc import ABC, abstractmethod
from fvhoe.hydro import HydroState
import numpy as np
from numbers import Number
from typing import Callable, Dict, Tuple, Union


_hs = HydroState(ndim=1)


class InitialCondition(ABC):
    """
    Abstract base class for building hydrodynamic initial conditions.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def base_ic(
        self, *args, **kwargs
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Abstract method for core hydrodynamic initial conditions, to be implemented in subclasses.
        Should return a function that produces an IC array with shape (5, nx, ny, nz).
        """
        pass

    def build_ic(
        self,
        hs: HydroState,
        passive_ic_funcs: Dict[
            str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        ] = None,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Combines base hydrodynamic IC function with passive scalar IC functions.
        Should return a function that produces an IC array with shape (5 + len(hs.passive_scalars), nx, ny, nz).
        args:
            hs (HydroState) : HydroState object
            passive_ic_funcs (Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) : dictionary of passive scalar IC functions
                each function must have the signature f(x, y, z) -> np.ndarray
                    args:
                        x (np.ndarray) : x-coordinate mesh, shape (nx, ny, nz)
                        y (np.ndarray) : y-coordinate mesh, shape (nx, ny, nz)
                        z (np.ndarray) : z-coordinate mesh, shape (nx, ny, nz)
                    returns:
                        out (np.ndarray) : array of passive scalar, shape (nx, ny, nz)
        returns:
            ic_func (Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) : function with signature
                args:
                    x (np.ndarray) : x-coordinate mesh, shape (nx, ny, nz)
                    y (np.ndarray) : y-coordinate mesh, shape (nx, ny, nz)
                    z (np.ndarray) : z-coordinate mesh, shape (nx, ny, nz)
                returns:
                    out (np.ndarray) : array of primitive variables, shape (5 + len(hs.passive_scalars), nx, ny, nz)
        """
        # check if all passive scalar ICs are provided
        if hs.includes_passives:
            missing_passives = hs.passive_scalars - passive_ic_funcs.keys()
            if missing_passives:
                raise ValueError(
                    f"Missing IC definitions for passive scalars: {missing_passives}"
                )

        # core hydrodynamic IC function
        core_ic_func = self.base_ic(*self.args, **self.kwargs)

        # combine core and passive scalar IC functions
        def ic_func(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            out = np.empty((5 + len(hs.passive_scalars), *x.shape))
            out[:5, ...] = core_ic_func(x, y, z)
            if hs.includes_passives:
                for scalar in hs.passive_scalars:
                    out[hs.variable_map[scalar], ...] = passive_ic_funcs[scalar](
                        x, y, z
                    )
            return out

        return ic_func


class Sinus(InitialCondition):
    def base_ic(
        self,
        dims: str = "x",
        rho_min_max: Tuple[float, float] = (1, 2),
        P: float = 1,
        vx: float = 1,
        vy: float = 0,
        vz: float = 0,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for a sinusoidal density wave
        args:
            dims (str) : contains "x", "y", and/or "z"
            rho_min_max (Tuple[float, float]) : min density, max density
            P (float) : uniform pressure
            vx (float) : uniform x-velocity
            vy (float) : uniform y-velocity
            vz (float) : uniform z-velocity
        returns:
            f(x, y, z)
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
            rho_min, rho_range = rho_min_max[0], rho_min_max[1] - rho_min_max[0]
            r = np.zeros_like(x)
            for dim, coord in zip("xyz", [x, y, z]):
                r += coord if dim in dims else 0

            out = np.empty((5,) + r.shape)
            out[_hs("rho")] = rho_range * (0.5 * np.sin(2 * np.pi * r) + 0.5) + rho_min
            out[_hs("vx")] = vx
            out[_hs("vy")] = vy
            out[_hs("vz")] = vz
            out[_hs("P")] = P
            return out

        return f


class Square(InitialCondition):
    def base_ic(
        self,
        dims: str = "x",
        rho_min_max: Tuple[float, float] = (1, 2),
        P: float = 1,
        vx: float = 1,
        vy: float = 0,
        vz: float = 0,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for a square density wave
        args:
            dims (str) : contains "x", "y", and/or "z"
            rho_min_max (Tuple[float, float]) : min density, max density
            P (float) : uniform pressure
            vx (float) : uniform x-velocity
            vy (float) : uniform y-velocity
            vz (float) : uniform z-velocity
        returns:
            f(x, y, z)
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
            inside_square = np.ones_like(x, dtype=bool)
            for dim, r in zip("xyz", [x, y, z]):
                inside_square &= (
                    np.logical_and(r > 0.25, r < 0.75) if dim in dims else True
                )

            out = np.empty((5,) + r.shape)
            out[_hs("rho")] = np.where(inside_square, rho_min_max[1], rho_min_max[0])
            out[_hs("vx")] = vx
            out[_hs("vy")] = vy
            out[_hs("vz")] = vz
            out[_hs("P")] = P
            return out

        return f


class SlottedDisk(InitialCondition):
    def base_ic(
        self,
        rho_min_max: Tuple[float, float] = (1, 2),
        P: float = 1,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for a slotted disk
        args:
            rho_min_max (Tuple[float, float]) : min density (outside disk), max density (inside disk)
            P (float) : uniform pressure
        returns:
            f(x, y, z)
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
            xc, yc = x - 0.5, y - 0.5
            rsq = np.square(xc) + np.square(y - 0.75)
            inside_disk = rsq < 0.15**2
            inside_disk &= np.logical_not(np.logical_and(np.abs(xc) < 0.025, y < 0.85))

            out = np.empty((5,) + rsq.shape)
            out[_hs("rho")] = np.where(inside_disk, rho_min_max[1], rho_min_max[0])
            out[_hs("vx")] = -yc
            out[_hs("vy")] = xc
            out[_hs("vz")] = 0
            out[_hs("P")] = P
            return out

        return f


class Shock1D(InitialCondition):
    def base_ic(
        self,
        dim: str = "x",
        position: float = 0.5,
        rho_left_right: Tuple[float, float] = (1, 0.125),
        P_left_right: Tuple[float, float] = (1, 0.1),
        v_left_right: Tuple[float, float] = (0, 0),
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for a normal shock tube
        args:
            dim (str) : "x", "y", or "z"
            position (float) : position of the shock
            rho_left_right (Tuple[float, float]) : density left and right of the shock
            P_left_right (Tuple[float, float]) : pressure left and right of the shock
            v_left_right (Tuple[float, float]) : velocity left and right of the shock
        returns:
            f(x, y, z)
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
            if dim not in "xyz":
                raise ValueError("dim must be 'x', 'y', or 'z'")
            r = {"x": x, "y": y, "z": z}[dim]
            v = np.where(r < position, v_left_right[0], v_left_right[1])

            out = np.empty((5,) + r.shape)
            out[_hs("rho")] = np.where(
                r < position, rho_left_right[0], rho_left_right[1]
            )
            out[_hs("vx")] = v if dim == "x" else 0
            out[_hs("vy")] = v if dim == "y" else 0
            out[_hs("vz")] = v if dim == "z" else 0
            out[_hs("P")] = np.where(r < position, P_left_right[0], P_left_right[1])
            return out

        return f


class DoubleShock1D(InitialCondition):
    def base_ic(
        self,
        dim: str = "x",
        positions: Tuple[float, float] = (0.1, 0.9),
        rhos: Tuple[float, float, float] = (1, 1, 1),
        vs: Tuple[float, float, float] = (0, 0, 0),
        Ps: Tuple[float, float, float] = (10**3, 10**-2, 10**2),
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for a double shock tube
        args:
            dim (str) : "x", "y", or "z"
            positions (Tuple[float, float]) : positions of the two shocks
            rhos (Tuple[float, float, float]) : densities left, between, and right of the shocks
            vs (Tuple[float, float, float]) : velocities left, between, and right of the shocks
            Ps (Tuple[float, float, float]) : pressures left, between, and right of the shocks
        returns:
            f(x, y, z)
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
            r = {"x": x, "y": y, "z": z}[dim]

            def place_shocks(r, values):
                return np.where(
                    r < positions[0],
                    values[0],
                    np.where(r < positions[1], values[1], values[2]),
                )

            out = np.empty((5,) + r.shape)
            out[_hs("rho")] = place_shocks(r, rhos)
            out[_hs("vx")] = place_shocks(r, vs) if dim == "x" else 0
            out[_hs("vy")] = place_shocks(r, vs) if dim == "y" else 0
            out[_hs("vz")] = place_shocks(r, vs) if dim == "z" else 0
            out[_hs("P")] = place_shocks(r, Ps)
            return out

        return f


class ShuOsher1D(InitialCondition):
    def base_ic(
        self,
        dim: str = "x",
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for the Shu-Osher problem
        args:
            dim (str) : "x", "y", or "z"
        returns:
            f(x, y, z)
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
            r = {"x": x, "y": y, "z": z}[dim] - 5
            v = v = np.where(r < -4, 2.629369, 0)

            out = np.empty((5,) + r.shape)
            out[_hs("rho")] = np.where(r < -4, 3.857143, 1 + 0.2 * np.sin(5 * r))
            out[_hs("vx")] = v if dim == "x" else 0
            out[_hs("vy")] = v if dim == "y" else 0
            out[_hs("vz")] = v if dim == "z" else 0
            out[_hs("P")] = np.where(r < -4, 10.33333, 1)
            return out

        return f


class KelvinHelmholtz2D(InitialCondition):
    def base_ic(
        self, sigma: float = 0.05 * np.sqrt(2), w0: float = 0.1
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for the Kelvin-Helmholtz instability
        args:
            sigma (float) : vy perturbation amplitude
            w0 (float) : vy perturbation exponential amplitude
        returns:
            f(x, y, z)
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
            inner_region = np.logical_and(0.25 < y, y < 0.75)

            out = np.empty((5,) + inner_region.shape)
            out[_hs("rho")] = np.where(inner_region, 2, 1)
            out[_hs("vx")] = np.where(inner_region, 0.5, -0.5)
            out[_hs("vy")] = (
                w0
                * np.sin(4 * np.pi * x)
                * (
                    np.exp(-np.square(y - 0.25) / (2 * sigma * sigma))
                    + np.exp(-np.square(y - 0.75) / (2 * sigma * sigma))
                )
            )
            out[_hs("vz")] = 0
            out[_hs("P")] = 2.5
            return out

        return f


class DoubleMachReflection2D(InitialCondition):
    def base_ic(
        self,
        gamma: float = 1.4,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for the double Mach reflection
        args:
            gamma (float) : specific heat ratio
        returns:
            f(x, y, z)
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
            xc = 1 / 6
            theta = np.pi / 3
            xp = x - y / np.tan(theta)

            out = np.empty((5,) + x.shape)
            out[_hs("rho")] = np.where(xp < xc, 8, gamma)
            out[_hs("vx")] = np.where(xp < xc, 7.145, 0)
            out[_hs("vy")] = np.where(xp < xc, -8.25 / 2, 0)
            out[_hs("vz")] = 0
            out[_hs("P")] = np.where(xp < xc, 116.5, 1)
            return out

        return f


class ShockTube(InitialCondition):
    def base_ic(
        self,
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
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for a multidimensional shock tube
        args:
            dims (str) : contains "x", "y", and/or "z"
            center (Union[float, Tuple[float]]) : center of the blast
            radius (float) : radius of the blast
            rho_in_out (Tuple[float, float]) : density inside and outside the blast
            P_in_out (Tuple[float, float]) : pressure inside and outside the blast
            vx_in_out (Tuple[float, float]) : x-velocity inside and outside the blast
            vy_in_out (Tuple[float, float]) : y-velocity inside and outside the blast
            vz_in_out (Tuple[float, float]) : z-velocity inside and outside the blast
            mode (str) : 'tube' or 'cube'. if 'cube', the blast is a rectangular prism and dims, center, and radius are ignored
            x_cube (Tuple[float, float]) : x-cube dimensions for mode='cube'
            y_cube (Tuple[float, float]) : y-cube dimensions for mode='cube'
            z_cube (Tuple[float, float]) : z-cube dimensions for mode='cube'
        returns:
            f(x, y, z)
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
            if mode == "tube":
                # convert single values to tuples
                center_as_tuple = (center,) if isinstance(center, Number) else center

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
            out = np.empty((5,) + x.shape)
            out[_hs("rho")] = np.where(inside_region, rho_in_out[0], rho_in_out[1])
            out[_hs("vx")] = np.where(inside_region, vx_in_out[0], vx_in_out[1])
            out[_hs("vy")] = np.where(inside_region, vy_in_out[0], vy_in_out[1])
            out[_hs("vz")] = np.where(inside_region, vz_in_out[0], vz_in_out[1])
            out[_hs("P")] = np.where(inside_region, P_in_out[0], P_in_out[1])
            return out

        return f


class SedovBlast(InitialCondition):
    def base_ic(
        self,
        mode: str = "corner",
        dims: str = "xy",
        rho0: float = 1.0,
        E0: float = 1e-5,
        E1: float = 1.0,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for the Sedov blast wave
        args:
            mode (str) : 'corner' or 'center'. if 'center', the blast is centered at the domain center
            dims (str) : contains "x", "y", and/or "z"
            rho0 (float) : background pressure
            E0 (float) : background energy
            E1 (float) : peak energy
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
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
            out = np.empty((5,) + x.shape)
            out[_hs("rho")] = rho0
            out[_hs("vx")] = 0
            out[_hs("vy")] = 0
            out[_hs("vz")] = 0
            out[_hs("P")] = E0

            # place peak energy based on mode
            if mode == "corner":
                peak = (0, 0, 0)
            elif mode == "center":
                peak = (
                    slice(None) if "x" not in dims else slice(Nx // 2 - 1, Nx // 2 + 1),
                    slice(None) if "y" not in dims else slice(Ny // 2 - 1, Ny // 2 + 1),
                    slice(None) if "z" not in dims else slice(Nz // 2 - 1, Nz // 2 + 1),
                )
            out[_hs("E")][peak] = Emax
            return out

        return f


class AthenaBlast(InitialCondition):
    def base_ic(
        self,
        mode: str = "corner",
        dims: str = "xy",
        rho0: float = 1.0,
        E0: float = 1e-5,
        E1: float = 1.0,
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        returns initial condition function for the Athena test blast https://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
        x in [-0.5, 0.5]
        y in [-0.75, 0.75]
        periodic boundary conditions
        """

        def f(
            x: np.ndarray,
            y: np.ndarray = None,
            z: np.ndarray = None,
        ) -> np.ndarray:
            # radius centered at (0, 0)
            r = np.sqrt(np.square(x) + np.square(y))

            # define initial conditions
            out = np.empty((5,) + r.shape)
            out[_hs("rho")] = 1
            out[_hs("vx")] = 0
            out[_hs("vy")] = 0
            out[_hs("vz")] = 0
            out[_hs("P")] = np.where(r < 0.1, 10, 0.1)
            return out

        return f
