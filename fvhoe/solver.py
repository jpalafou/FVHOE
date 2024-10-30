from fvhoe.array_management import get_array_slice as slc
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.fv import (
    conservative_interpolation,
    fv_average,
    fv_uniform_meshgen,
    interpolate_cell_centers,
    interpolate_fv_averages,
    transverse_reconstruction,
)
from fvhoe.hydro import compute_conservatives, compute_primitives, hydro_dt, HydroState
from fvhoe.initial_conditions import InitialCondition, Square
from fvhoe.ode import ODE
from fvhoe.riemann_solvers import advection_upwind, hllc, llf
from fvhoe.slope_limiting import (
    broadcast_to_troubled_interfaces,
    detect_troubled_cells,
    MUSCL_interpolations,
)
from fvhoe.stencils import get_stencil_size
from fvhoe.visualization import plot_1d_slice, plot_2d_slice
import inspect
from itertools import product
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
from typing import Callable, Dict, Iterable, Tuple

try:
    from cupy import ndarray as cp_ndarray

    CUPY_AVAILABLE = True
except Exception:
    from numpy import ndarray as cp_ndarray

    CUPY_AVAILABLE = False


class EulerSolver(ODE):
    def __init__(
        self,
        w0: InitialCondition = Square(),
        w0_passives: Dict[
            str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        ] = None,
        nx: int = 1,
        ny: int = 1,
        nz: int = 1,
        px: int = 0,
        py: int = 0,
        pz: int = 0,
        x: Tuple[float, float] = (0, 1),
        y: Tuple[float, float] = (0, 1),
        z: Tuple[float, float] = (0, 1),
        CFL: float = 0.8,
        fixed_dt: float = None,
        gamma: float = 5 / 3,
        bc: BoundaryCondition = None,
        riemann_solver: str = "hllc",
        conservative_ic: bool = False,
        fv_ic: bool = False,
        fixed_primitive_variables: Iterable = None,
        a_posteriori_slope_limiting: bool = False,
        slope_limiter: str = "minmod",
        force_trouble: bool = False,
        NAD: float = 1e-2,
        NAD_mode: str = "global",
        NAD_range: str = "relative",
        NAD_vars: tuple = None,
        PAD: dict = None,
        SED: bool = True,
        SED_tolerance: float = 1e-10,
        convex: bool = False,
        density_floor: float = None,
        pressure_floor: float = None,
        csq_floor: float = 1e-10,
        progress_bar: bool = True,
        snapshots_as_fv_averages: bool = True,
        snapshot_helper_function: callable = None,
        slab_buffer_size: int = 30,
        cupy: bool = False,
    ):
        """
        solver for Euler equations, a system of 5+ variables:
            rho (density)
            P (pressure)
            vx (x-velocity)
            vy (y-velocity)
            vz (z-velocity)
            passive_scalar1, passive_scalar2, ... (optional)
        implemented in 1D, 2D, and 3D
        args:
            w0 (InitialCondition) : initial condition function generator
            w0_passives (Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) : dictionary of passive scalar IC functions
                each function must have the signature f(x, y, z) -> np.ndarray
                    args:
                        x (np.ndarray) : x-coordinate mesh, shape (nx, ny, nz)
                        y (np.ndarray) : y-coordinate mesh, shape (nx, ny, nz)
                        z (np.ndarray) : z-coordinate mesh, shape (nx, ny, nz)
                    returns:
                        out (np.ndarray) : array of passive scalar, shape (nx, ny, nz)
            nx (int) : number of cells along x-direction. ignore by setting nx=1, px=0
            ny (int) : number of cells along y-direction. ignore by setting ny=1, py=0
            nz (int) : number of cells along z-direction. ignore by setting nz=1, pz=0
            px (int) : polynomial degree along x-direction
            py (int) : polynomial degree along y-direction
            pz (int) : polynomial degree along z-direction
            x (Tuple[float, float]) : x domain bounds (x1, x2)
            y (Tuple[float, float]) : y domain bounds (y1, y2)
            z (Tuple[float, float]) : z domain bounds (z1, z2)
            CFL (float) : Courant-Friedrichs-Lewy condition
            fixed_dt (float) : ignore CFL and use this constant time-step size if it isn't defined as None
            gamma (float) : specific heat ratio
            bc (BoundaryCondition) : boundary condition instance for primitive variables
                None : apply periodic boundaries
            riemann_solver (str) : riemann solver code
                "advection_upwinding" : for pure advection problem
                "llf" : simple riemann solver for Euler equations
                "hllc" : advanced riemann solver for Euler equations
            conservative_ic (bool) : indicates that w0 returns conservative variables if true
            fv_ic (bool) : indicates that w0 returns finite volume averages if true
            fixed_primitive_variables (Iterable) : series of primitive variables to keep fixed to their initial value
            a_posteriori_slope_limiting (bool) : whether to apply a postreiori slope limiting
            slope_limiter (str) : slope limiter code, "minmod", "moncen", None
            force_trouble (bool) : if True, all cells are flagged as troubled
            NAD (float) : NAD tolerance in troubled cell detection
            NAD_mode (str) : "global" or "local"
                "global" : NAD is applied based on the global range of each variable
                "local" : NAD is applied based on the local range of each variable
            NAD_range (str) : "relative" or "absolute"
                "relative" : NAD is applied based on the relative range of each variable
                    upper_bound = max + (max - min) * eps
                    lower_bound = min - (max - min) * eps
                "absolute" : NAD is applied based on the absolute range of each variable
                    upper_bound = (1 + eps) * max
                    lower_bound = (1 - eps) * min
            NAD_vars (tuple) : tuple of variables to apply NAD. if None, all variables are considered
            PAD (dict) : primitive variable limits for slope limiting
            SED (bool) : whether to ignore NAD trouble where smooth extrema are detected
            SED_tolerance (float) : tolerance for avoiding dividing by 0 in smooth extrema detection
            convex (bool) : whether to apply convex slope limiting
            density_floor (float) : density floor after conservative interpolation. if None, a floor is not applied
            pressure_floor (float) : pressure floor after conservative interpolation. if None, a floor is not applied
            csq_floor (float) : floor on square of returned sound speed
            progress_bar (bool) : whether to print out a progress bar
            snapshots_as_fv_averages (bool) : save snapshots as finite volume averages. if false, save as cell centers
            snapshot_helper_function (callable) : function to call at the end of a snapshot with self as the sole argument
            slab_buffer_size (int) : for applying boundary conditions
            cupy (bool) : whether to use GPUs via the cupy library
        returns:
            EulerSolver object
        """
        self.csq_floor = csq_floor
        self.density_floor = density_floor
        self.gamma = gamma
        self.pressure_floor = pressure_floor
        self.riemann_solver_name = riemann_solver
        self.snapshot_helper_function = snapshot_helper_function
        self.snapshots_as_fv_averages = snapshots_as_fv_averages

        # call init functions
        self._init_mesh(x, y, z, nx, ny, nz, px, py, pz, CFL, fixed_dt)
        self._init_initial_conditions(
            w0, w0_passives, conservative_ic, fv_ic, progress_bar, cupy
        )
        self._init_boundary_conditions(slab_buffer_size, bc, fv_ic)
        self._init_slope_limiting(
            a_posteriori_slope_limiting,
            slope_limiter,
            force_trouble,
            NAD,
            NAD_mode,
            NAD_range,
            NAD_vars,
            PAD,
            SED,
            SED_tolerance,
            convex,
        )
        self._init_array_allocation()

        # configure riemann solver
        match self.riemann_solver_name:
            case "advection_upwind":
                self.riemann_solver = advection_upwind
            case "llf":
                self.riemann_solver = llf
            case "hllc":
                self.riemann_solver = hllc
            case _:
                raise TypeError(f"Invalid Riemann solver {riemann_solver}")

        # configure plotting functions
        self.plot_1d_slice = lambda *args, **kwargs: plot_1d_slice(
            self, *args, **kwargs
        )
        self.plot_2d_slice = lambda *args, **kwargs: plot_2d_slice(
            self, *args, **kwargs
        )

        # configure timer categories
        self.timer.add_cat(
            [
                "boundary conditions",
                "(high-order) hydrofluxes",
                "(high-order) riemann solver",
                "(high-order) conservative interpolation",
                "(high-order) transverse reconstruction",
                "(fallback scheme)",
                "(fallback scheme) troubled cell detection",
                "(fallback scheme) hydrofluxes",
                "(fallback scheme) riemann solver",
                "(fallback scheme) conservative interpolation",
                "(fallback scheme) transverse reconstruction",
            ]
        )

        # configure variable slicer

    def _init_mesh(self, x, y, z, nx, ny, nz, px, py, pz, CFL, fixed_dt):
        """
        initialize mesh parameters
        """

        self.CFL = CFL
        self.fixed_dt = fixed_dt

        # generate txyz mesh
        self.x_domain = x
        self.y_domain = y
        self.z_domain = z
        hx = (x[1] - x[0]) / nx
        hy = (y[1] - y[0]) / ny
        hz = (z[1] - z[0]) / nz
        self.h = (hx, hy, hz)
        self.n = (nx, ny, nz)
        self.p = (px, py, pz)
        self.X, self.Y, self.Z = fv_uniform_meshgen(
            self.n, self.x_domain, self.y_domain, self.z_domain
        )
        self.x = self.X[:, 0, 0]
        self.y = self.Y[0, :, 0]
        self.z = self.Z[0, 0, :]
        self.xdim = not (self.n[0] == 1 and self.p[0] == 0)
        self.ydim = not (self.n[1] == 1 and self.p[1] == 0)
        self.zdim = not (self.n[2] == 1 and self.p[2] == 0)
        self.dims = ""
        if self.xdim:
            self.dims += "x"
        if self.ydim:
            self.dims += "y"
        if self.zdim:
            self.dims += "z"
        self.ndim = len(self.dims)

    def _init_initial_conditions(
        self, w0, w0_passives, conservative_ic, fv_ic, progress_bar, cupy
    ):
        """
        initialize initial conditions, integrator, and array manager
        """
        self.conservative_ic = conservative_ic
        self.fv_ic = fv_ic

        # define hydro state
        passive_scalars = tuple(w0_passives.keys()) if w0_passives is not None else ()
        self.hydro_state = HydroState(passive_scalars=passive_scalars)
        _hs = self.hydro_state
        self.nvars = _hs.nvars

        # user-defined initial condition function
        ic_func = w0.build_ic(self.hydro_state, w0_passives)

        # check if w0 accepts x, y, z as arguments
        if not {"x", "y", "z"}.issubset(inspect.signature(ic_func).parameters):
            raise ValueError(
                "Initial condition function must accept x, y, z as arguments."
            )

        # define the following initial condition functions (if possible)
        #   w0(X, Y, Z) : pointwise primitive variables
        #   u0(X, Y, Z) : pointwise conservative variables
        #   u0_fv(X, Y, Z) : finite volume averages of conservative variables
        if self.conservative_ic and self.fv_ic:
            self.w0 = NotImplemented
            self.u0 = NotImplemented
            self.u0_fv = ic_func
        elif self.fv_ic:
            raise ValueError(
                "Can't have finite volume averages without conservative IC."
            )
        else:
            if self.conservative_ic:
                self.w0 = NotImplemented
                self.u0 = ic_func
            else:
                self.w0 = ic_func
                self.u0 = lambda x, y, z: compute_conservatives(
                    _hs, ic_func(x, y, z), gamma=self.gamma
                )
            self.u0_fv = lambda x, y, z: fv_average(
                self.u0, x, y, z, h=self.h, p=self.p
            )

        # array of finite volume averages
        u0_fv_arr = self.u0_fv(self.X, self.Y, self.Z)

        # inegrator class init
        super().__init__(u0_fv_arr, progress_bar=progress_bar, cupy=cupy)

        # initialize f evaluation count
        self.f_evaluation_count = 0

    def _init_boundary_conditions(self, slab_buffer_size, bc, fv_ic):
        """
        initialize boundary conditions
        """
        self.slab_buffer_size = slab_buffer_size

        # get slab coordinates for boundaries
        slab_buffer_sizes = (
            self.slab_buffer_size if self.xdim else 0,
            self.slab_buffer_size if self.ydim else 0,
            self.slab_buffer_size if self.zdim else 0,
        )
        _, slab_coords = fv_uniform_meshgen(
            (self.n[0], self.n[1], self.n[2]),
            x=self.x_domain,
            y=self.y_domain,
            z=self.z_domain,
            slab_thickness=slab_buffer_sizes,
        )

        # define boundary conditions
        if bc is None:
            self.bc = BoundaryCondition(
                x=("periodic", "periodic"),
                y=("periodic", "periodic"),
                z=("periodic", "periodic"),
                array_manager=self.am,
            )
        else:
            self.bc = BoundaryCondition(
                x=bc.x,
                y=bc.y,
                z=bc.z,
                x_value=bc.x_value,
                y_value=bc.y_value,
                z_value=bc.z_value,
                slab_coords=slab_coords,
                array_manager=self.am,
            )

        # configure initial condition boundary conditions
        if "ic" in set(self.bc.x + self.bc.y + self.bc.z):
            for dim in "xyz":
                if getattr(self.bc, dim)[0] == "ic":
                    self.bc.reset_value(dim, "l", self.u0)
                if getattr(self.bc, dim)[1] == "ic":
                    self.bc.reset_value(dim, "r", self.u0)

    def _init_slope_limiting(
        self,
        a_posteriori_slope_limiting,
        slope_limiter,
        force_trouble,
        NAD,
        NAD_mode,
        NAD_range,
        NAD_vars,
        PAD,
        SED,
        SED_tolerance,
        convex,
    ):
        """
        initialize slope limiting parameters
        """
        self.a_posteriori_slope_limiting = a_posteriori_slope_limiting
        self.slope_limiter = slope_limiter
        self.force_trouble = force_trouble
        self.NAD = NAD
        self.NAD_mode = NAD_mode
        self.NAD_range = NAD_range
        self.NAD_vars = (
            NAD_vars
            if isinstance(NAD_vars, tuple) or NAD_vars is None
            else tuple(NAD_vars)
        )
        default_PAD = {
            "rho": (0.0, np.inf),
            "P": (0.0, np.inf),
        }
        self.PAD = default_PAD if PAD is None else PAD
        self.SED = SED
        self.SED_tolerance = SED_tolerance
        self.convex = convex
        self.trouble_counter = 1

    def _init_array_allocation(self):
        """
        allocate Numpy or CuPy arrays
        """
        nvars = self.nvars
        nx, ny, nz = self.n

        # primitive variables
        self.am.add("w", np.empty((nvars, nx, ny, nz)))

        # limiting
        self.am.add("mean trouble", np.zeros((5, nx, ny, nz)))
        self.am.add("mean NAD mag", np.zeros((5, nx, ny, nz)))
        self.am.add("mean PAD mag", np.zeros((5, nx, ny, nz)))

        # fluxes
        self.am.add("F", np.empty((nvars, nx + 1, ny, nz)))
        self.am.add("G", np.empty((nvars, nx, ny + 1, nz)))
        self.am.add("H", np.empty((nvars, nx, ny, nz + 1)))

    def f(self, t, u):
        """
        compute the RHS of the Euler equations
        """
        self.f_evaluation_count += 1
        return self.hydrodynamics(u)

    def hydrodynamics(self, u: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        compute the Euler equation dynamics from cell-average conserved values u
        args:
            u (array_like) : fv averages of conservative variables. has shape (nvars, nx, ny, nz)
        returns:
            dt, dudt (float, array_like) : time-step size, conservative variable dynamics
        """
        # high-order fluxes
        dt, (self.am("F")[...], self.am("G")[...], self.am("H")[...]) = (
            self.hydrofluxes(u=u, p=self.p, timer_prefix="(high-order) ")
        )

        if self.a_posteriori_slope_limiting:
            self.revise_fluxes(
                u=u,
                F=self.am("F"),
                G=self.am("G"),
                H=self.am("H"),
                dt=dt,
                force_trouble=self.force_trouble,
            )

        # compute conservative variable dynamics
        dudt = self.euler_equation(F=self.am("F"), G=self.am("G"), H=self.am("H"))
        return dt, dudt

    def hydrofluxes(
        self,
        u: np.ndarray,
        p: Tuple[int, int, int],
        slope_limiter: str = None,
        timer_prefix: str = "",
    ) -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        compute the Euler equation fluxes F, G, H with degree p polynomials. optionally apply slope limiting.
        if xdim is False, F is returned as self.F and so on for G and H
        args:
            u (array_like) : cell-averages of conservative variables. has shape (nvars, nx, ny, nz)
            p (Tuple[int, int, int]) : polynomial interpolation degree (px, py, pz)
            limiter (str) : None, 'minmod', 'moncen'
            timer_prefix (str) : prefix for timer categories
        returns:
            dt, (F, G, H) (tuple) : time-step size and numerical fluxes
                dt (float) : time-step size
                F (array_like) : x-direction conservative fluxes. has shape (nvars, nx + 1, ny, nz)
                G (array_like) : y-direction conservative fluxes. has shape (nvars, nx, ny + 1, nz)
                H (array_like) : z-direction conservative fluxes. has shape (nvars, nx, ny, nz + 1)
        """
        self.timer.start(f"{timer_prefix}hydrofluxes")
        _hs = self.hydro_state

        # determine if slope limiting is required
        if slope_limiter is None:
            limit_slopes = False
        elif max(p) <= 1:
            limit_slopes = True
        else:
            raise ValueError(
                "Slope limiting is only implemented for degree 1 polynomial interpolation."
            )

        # find required number of ghost zones
        pmax = max(p)
        gw = max(int(np.ceil(pmax / 2)) + 1, 2 * int(np.ceil(pmax / 2)))

        # fv conservatives to fv primitives
        w_bc = self.interpolate_primitives_from_conservatives(
            u=u,
            p=p,
            gw=(
                gw if self.xdim else 0,
                gw if self.ydim else 0,
                gw if self.zdim else 0,
            ),
        )

        # hard-coded floors
        if self.density_floor is not None:
            w_bc[slc("rho")] = np.maximum(w_bc[slc("rho")], self.density_floor)
        if self.pressure_floor is not None:
            w_bc[slc("P")] = np.maximum(w_bc[slc("P")], self.pressure_floor)

        # and time-step size
        if self.fixed_dt is None:
            dt = hydro_dt(
                w=w_bc,
                h=min(self.h),
                ndim=self.ndim,
                CFL=self.CFL,
                gamma=self.gamma,
                csq_floor=self.csq_floor,
            )
        else:
            dt = self.fixed_dt

        # perform conservative interpolation
        self.timer.start(f"{timer_prefix}conservative interpolation")

        # interpolate x face midpoints
        if self.xdim:
            w_xy = conservative_interpolation(w_bc, p=p[2], axis=3, pos="c")
            w_x = conservative_interpolation(w_xy, p=p[1], axis=2, pos="c")
            if limit_slopes and p[0] == 1:
                w_x_face_center_l, w_x_face_center_r = MUSCL_interpolations(
                    w_x, axis=1, limiter=slope_limiter
                )
            else:
                w_x_face_center_l = conservative_interpolation(
                    w_x, p=p[0], axis=1, pos="l"
                )
                w_x_face_center_r = conservative_interpolation(
                    w_x, p=p[0], axis=1, pos="r"
                )

        # interpolate y face midpoints
        if self.ydim:
            w_yz = conservative_interpolation(w_bc, p=p[0], axis=1, pos="c")
            w_y = conservative_interpolation(w_yz, p=p[2], axis=3, pos="c")
            if limit_slopes and p[1] == 1:
                w_y_face_center_l, w_y_face_center_r = MUSCL_interpolations(
                    w_y, axis=2, limiter=slope_limiter
                )
            else:
                w_y_face_center_l = conservative_interpolation(
                    w_y, p=p[1], axis=2, pos="l"
                )
                w_y_face_center_r = conservative_interpolation(
                    w_y, p=p[1], axis=2, pos="r"
                )

        # interpolate z face midpoints
        if self.zdim:
            w_zx = conservative_interpolation(w_bc, p=p[1], axis=2, pos="c")
            w_z = conservative_interpolation(w_zx, p=p[0], axis=1, pos="c")
            if limit_slopes and p[2] == 1:
                w_z_face_center_l, w_z_face_center_r = MUSCL_interpolations(
                    w_z, axis=3, limiter=slope_limiter
                )
            else:
                w_z_face_center_l = conservative_interpolation(
                    w_z, p=p[2], axis=3, pos="l"
                )
                w_z_face_center_r = conservative_interpolation(
                    w_z, p=p[2], axis=3, pos="r"
                )

        self.timer.stop(f"{timer_prefix}conservative interpolation")

        # pointwise numerical fluxes
        self.timer.start(f"{timer_prefix}riemann solver")
        if self.xdim:
            f_face_center = self.riemann_solver(
                hs=_hs,
                riemann_problem=(
                    w_x_face_center_r[:, :-1, ...],
                    w_x_face_center_l[:, 1:, ...],
                ),
                gamma=self.gamma,
                dim="x",
                csq_floor=self.csq_floor,
            )
        if self.ydim:
            g_face_center = self.riemann_solver(
                hs=_hs,
                riemann_problem=(
                    w_y_face_center_r[:, :, :-1, ...],
                    w_y_face_center_l[:, :, 1:, ...],
                ),
                gamma=self.gamma,
                dim="y",
                csq_floor=self.csq_floor,
            )
        if self.zdim:
            h_face_center = self.riemann_solver(
                hs=_hs,
                riemann_problem=(
                    w_z_face_center_r[:, :, :, :-1, ...],
                    w_z_face_center_l[:, :, :, 1:, ...],
                ),
                gamma=self.gamma,
                dim="z",
                csq_floor=self.csq_floor,
            )
        self.timer.stop(f"{timer_prefix}riemann solver")

        # excess ghost zone counts after flux integral
        x_excess = (
            0,
            gw - int(np.ceil(p[0] / 2)) - 1 if self.xdim else 0,
            gw - 2 * int(np.ceil(p[1] / 2)) if self.ydim else 0,
            gw - 2 * int(np.ceil(p[2] / 2)) if self.zdim else 0,
        )
        y_excess = (
            0,
            gw - 2 * int(np.ceil(p[0] / 2)) if self.xdim else 0,
            gw - int(np.ceil(p[1] / 2)) - 1 if self.ydim else 0,
            gw - 2 * int(np.ceil(p[2] / 2)) if self.zdim else 0,
        )
        z_excess = (
            0,
            gw - 2 * int(np.ceil(p[0] / 2)) if self.xdim else 0,
            gw - 2 * int(np.ceil(p[1] / 2)) if self.ydim else 0,
            gw - int(np.ceil(p[2] / 2)) - 1 if self.zdim else 0,
        )

        # flux integrals
        self.timer.start(f"{timer_prefix}transverse reconstruction")
        if self.xdim:
            F = transverse_reconstruction(
                transverse_reconstruction(f_face_center, axis=2, p=p[1]), axis=3, p=p[2]
            )[tuple(slice(x_excess[i] or None, -x_excess[i] or None) for i in range(4))]
        else:
            F = self.am("F")
        if self.ydim:
            G = transverse_reconstruction(
                transverse_reconstruction(g_face_center, axis=1, p=p[0]), axis=3, p=p[2]
            )[tuple(slice(y_excess[i] or None, -y_excess[i] or None) for i in range(4))]
        else:
            G = self.am("G")
        if self.zdim:
            H = transverse_reconstruction(
                transverse_reconstruction(h_face_center, axis=1, p=p[0]), axis=2, p=p[1]
            )[tuple(slice(z_excess[i] or None, -z_excess[i] or None) for i in range(4))]
        else:
            H = self.am("H")
        self.timer.stop(f"{timer_prefix}transverse reconstruction")
        self.timer.stop(f"{timer_prefix}hydrofluxes")
        return dt, (F, G, H)

    def revise_fluxes(
        self,
        u: np.ndarray,
        F: np.ndarray,
        G: np.ndarray,
        H: np.ndarray,
        dt: float,
        force_trouble: bool = False,
    ) -> None:
        """
        revise fluxes to prevent oscillations
        args:
            u (array_like) : fv averages of conservative variables. has shape (nvars, nx, ny, nz)
            F (array_like) : x-fluxes. has shape (nvars, nx + 1, ny, nz)
            G (array_like) : y-fluxes. has shape (nvars, nx, ny + 1, nz)
            H (array_like) : z-fluxes. has shape (nvars, nx, ny, nz + 1)
            dt (float) : time-step size
            force_trouble (bool) : all cells are troubled
        returns:
            None : revise fluxes in place
        """
        self.timer.start("(fallback scheme)")
        _hs = self.hydro_state

        # compute candidate solution
        ustar = u + dt * self.euler_equation(F=F, G=G, H=H)

        # interpolate finite volume primitives from finite volume conservatives
        gw = (
            3 if self.xdim else 0,
            3 if self.ydim else 0,
            3 if self.zdim else 0,
        )
        w = self.interpolate_primitives_from_conservatives(u, p=self.p, gw=gw)
        w_star = self.interpolate_primitives_from_conservatives(ustar, p=self.p, gw=gw)

        # detect troubled cells
        self.timer.start("(fallback scheme) troubled cell detection")
        troubled_cells, NAD_mag, PAD_mag = detect_troubled_cells(
            u=w[_hs("active_scalars")],
            u_candidate=w_star[_hs("active_scalars")],
            dims=self.dims,
            NAD_eps=self.NAD,
            mode=self.NAD_mode,
            range_type=self.NAD_range,
            NAD_vars=self.NAD_vars,
            PAD_bounds=self.PAD,
            SED=self.SED,
            SED_eps=self.SED_tolerance,
            xp={True: "cupy", False: "numpy"}[self.am.using_cupy],
        )
        self.timer.stop("(fallback scheme) troubled cell detection")
        if force_trouble:
            troubled_cells = np.ones_like(troubled_cells, dtype=bool)

        self.log_troubles(troubled_cells, NAD_mag, PAD_mag)

        if np.any(troubled_cells):
            # p=1, slope-limited fluxes
            _, (Fl, Gl, Hl) = self.hydrofluxes(
                u=u,
                p=(int(self.p[0] > 0), int(self.p[1] > 0), int(self.p[2] > 0)),
                slope_limiter=self.slope_limiter,
                timer_prefix="(fallback scheme) ",
            )

            # get mask for troubled interfaces
            troubled_x, troubled_y, troubled_z = broadcast_to_troubled_interfaces(
                troubled_cells,
                dims=self.dims,
                convex=self.convex,
                periodic_x=self.bc.x == ("periodic", "periodic"),
                periodic_y=self.bc.y == ("periodic", "periodic"),
                periodic_z=self.bc.z == ("periodic", "periodic"),
                xp={True: "cupy", False: "numpy"}[self.am.using_cupy],
            )
            self.am("F")[...] = (1 - troubled_x) * F + troubled_x * Fl
            self.am("G")[...] = (1 - troubled_y) * G + troubled_y * Gl
            self.am("H")[...] = (1 - troubled_z) * H + troubled_z * Hl
        self.timer.stop("(fallback scheme)")

    def euler_equation(self, F: np.ndarray, G: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        compute the Euler equation dynamics from fv-average conserved values u
        args:
            F (array_like) : x-fluxes. has shape (nvars, nx + 1, ny, nz)
            G (array_like) : y-fluxes. has shape (nvars, nx, ny + 1, nz)
            H (array_like) : z-fluxes. has shape (nvars, nx, ny, nz + 1)
        returns:
            dudt (array_like) : conservative variable dynamics
        """
        dudt = 0.0
        if self.xdim:
            dudt += -(1 / self.h[0]) * (F[:, 1:, ...] - F[:, :-1, ...])
        if self.ydim:
            dudt += -(1 / self.h[1]) * (G[:, :, 1:, ...] - G[:, :, :-1, ...])
        if self.zdim:
            dudt += -(1 / self.h[2]) * (H[:, :, :, 1:, ...] - H[:, :, :, :-1, ...])
        return dudt

    def interpolate_primitives_from_conservatives(
        self,
        u: np.ndarray,
        p: Tuple[int, int, int],
        gw: Tuple[int, int, int] = (0, 0, 0),
        fv_average: bool = True,
    ) -> np.ndarray:
        """
        interpolate finite volume primitives from finite volume conservatives
        args:
            u (array_like) : finite volume conservative variables
            p (Tuple[int, int, int]) : interpolation polynomial degree in 3D (px, py, pz)
            gw (Tuple[int, int, int]) : how many ghost zones w should have on each side (gx, gy, gz).
                all values should be at least 0
            fv_average (bool) : whether u is cell-averaged or centroids
        returns:
            w (array_like) : primitive variables as finite volume averages or centroids
        """
        _hs = self.hydro_state

        # find required number of ghost zones
        if fv_average:
            interp_cost_x = 2 * get_stencil_size(p[0])
            interp_cost_y = 2 * get_stencil_size(p[1])
            interp_cost_z = 2 * get_stencil_size(p[2])
        else:
            interp_cost_x = get_stencil_size(p[0])
            interp_cost_y = get_stencil_size(p[1])
            interp_cost_z = get_stencil_size(p[2])
        gw = (gw[0] + interp_cost_x, gw[1] + interp_cost_y, gw[2] + interp_cost_z)
        self.timer.start("boundary conditions")
        u_bc = self.bc.apply(u, gw=gw, t=self.t)
        self.timer.stop("boundary conditions")
        u_cell_centers = interpolate_cell_centers(u_bc, p=p)
        w_cell_centers = compute_primitives(_hs, u_cell_centers, gamma=self.gamma)
        if fv_average:
            w = interpolate_fv_averages(w_cell_centers, p=p)
        else:
            w = w_cell_centers
        return w

    def log_troubles(
        self,
        trouble: np.ndarray = None,
        NAD_violation_magnitude: np.ndarray = None,
        PAD_violation_magnitude: np.ndarray = None,
        reset: bool = False,
    ) -> None:
        """
        log troubled cells and PAD violation magnitude as the mean along each Runge-Kutta substep
        args:
            trouble (array_like) : array of troubled cells
            NAD_violation_magnitude (array_like) : array of PAD violation magnitudes
            PAD_violation_magnitude (array_like) : array of PAD violation magnitudes
            reset (bool) : reset the trouble counter. if true, ignores all other arguments
        """
        if reset and self.trouble_counter > 0:
            self.am("mean trouble")[...] = (
                self.am("mean trouble") / self.trouble_counter
            )
            self.am("mean NAD mag")[...] = (
                self.am("mean NAD mag") / self.trouble_counter
            )
            self.am("mean PAD mag")[...] = (
                self.am("mean PAD mag") / self.trouble_counter
            )
            self.trouble_counter = 0
            return
        if self.trouble_counter == 0:
            self.am("mean trouble")[...] = trouble
            self.am("mean NAD mag")[...] = NAD_violation_magnitude
            self.am("mean PAD mag")[...] = PAD_violation_magnitude
        else:
            self.am("mean trouble")[...] = self.am("mean trouble") + trouble
            self.am("mean NAD mag")[...] = (
                self.am("mean NAD mag") + NAD_violation_magnitude
            )
            self.am("mean PAD mag")[...] = (
                self.am("mean PAD mag") + PAD_violation_magnitude
            )
        self.trouble_counter += 1

    def step_helper_function(self):
        # log
        if self.a_posteriori_slope_limiting:
            self.log_troubles(reset=True)

    def snapshot(self):
        """
        log a dictionary
        """
        # fv conservatives to fv primitives
        self.am("w")[...] = self.interpolate_primitives_from_conservatives(
            u=self.am("u"),
            p=self.p,
            fv_average=self.snapshots_as_fv_averages,
        )

        # log troubles
        if self.a_posteriori_slope_limiting:
            trouble = self.am.get_numpy("mean trouble")
            NAD_mag = self.am.get_numpy("mean NAD mag")
            PAD_mag = self.am.get_numpy("mean PAD mag")

        # append dictionary to list of snapshots
        log = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "t": self.t,
            "w": self.am.get_numpy("w"),
        }
        if self.a_posteriori_slope_limiting:
            log["trouble"] = trouble
            log["NAD violation magnitude"] = NAD_mag
            log["PAD violation magnitude"] = PAD_mag

        self.snapshots.append(log)
        self.snapshot_times.append(self.t)

        if self.snapshot_helper_function is not None:
            self.snapshot_helper_function(self)

    def read_snapshots(self) -> bool:
        """
        read snapshots from a pickle file
        returns:
            out (bool) : whether a file was found
        """
        if os.path.exists(self.snapshot_dir):
            with open(os.path.join(self.snapshot_dir, "arrs.pkl"), "rb") as f:
                self.snapshots = pickle.load(f)
            print(f"Read from snapshot directory {self.snapshot_dir}")
            return True
        return False

    def write_snapshots(self, overwrite: bool):
        """
        write snapshots to a pickle file
        args:
            overwrite (bool) : overwrite the file if it exists
        """
        snapshot_dir = self.snapshot_dir

        if os.path.exists(snapshot_dir) and not overwrite:
            raise FileExistsError(f"Snapshot directory {snapshot_dir} already exists.")
        elif os.path.exists(snapshot_dir) and overwrite:
            # Clear out the snapshot directory
            shutil.rmtree(snapshot_dir)
            print(f"Clearing out snapshot directory {snapshot_dir}")
        os.makedirs(snapshot_dir)

        # Write the snapshots to a pickle file
        with open(os.path.join(snapshot_dir, "arrs.pkl"), "wb") as f:
            pickle.dump(self.snapshots, f)

        # Save the rest of the attributes (excluding functions, etc.) as a json
        attrs_to_save = {}
        for k, v in self.__dict__.items():
            # save name of initial condition function
            if k == "w0":
                attrs_to_save[k] = getattr(v, "__name__", "__name__ not found")
                continue
            # skip functions, arrays, etc
            if (
                (callable(v) and k != "am")
                or isinstance(v, np.ndarray)
                or isinstance(v, cp_ndarray)
                or k
                in [
                    "progress_bar",
                    "timestamps",
                    "snapshots",
                    "w0_cell_centers_cache",
                ]
            ):
                continue
            # save dicts of array manager, boundary conditions, and timer
            if k in ["am", "bc", "timer"]:
                attrs_to_save[k] = v.to_dict()
                continue
            # rewrite np.inf as a string
            if k == "PAD":
                attrs_to_save[k] = {
                    k: [{np.inf: "inf", -np.inf: "-inf"}.get(item, item) for item in v]
                    for k, v in v.items()
                }
                continue
            attrs_to_save[k] = v

        # Write the attributes to a json file
        sorted_keys = sorted(attrs_to_save.keys(), key=lambda x: (x.lower(), x))
        attrs_to_save = {key: attrs_to_save[key] for key in sorted_keys}
        with open(os.path.join(snapshot_dir, "attrs.json"), "w") as f:
            json.dump(attrs_to_save, f, indent=4)

        # generate timing report
        with open(os.path.join(snapshot_dir, "timing.txt"), "w") as f:
            f.write(self.timer.report())

        print(f"Wrote to snapshot directory {snapshot_dir}")

    def run(self, *args, **kwargs):
        """
        solve forward in time using a Runge-Kutta method whose order matches the
            chosen polynomial degree for the spatial discretization, up to RK4
        args:
            *args : arguments to pass to the Runge-Kutta method
            **kwargs : keyword arguments to pass to the Runge-Kutta method
        """
        q = min(max(self.p), 3)
        match q:
            case 0:
                self.euler(*args, **kwargs)
            case 1:
                self.ssprk2(*args, **kwargs)
            case 2:
                self.ssprk3(*args, **kwargs)
            case 3:
                self.rk4(*args, **kwargs)
            case _:
                raise ValueError(f"Runge-Kutta method not implemented for {q=}")

    def plot_fields(self, **kwargs):
        if self.ndim == 1:
            fig, axs = plt.subplots(1, 3, sharex=True, figsize=(15, 5))
            self.plot_1d_slice(axs[0], param="rho", **kwargs)
            self.plot_1d_slice(axs[1], param="P", **kwargs)
            self.plot_1d_slice(axs[2], param="v" + self.dims, **kwargs)
            return fig, axs
        if self.ndim == 2:
            fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 9))
            varnames = ["rho", "P", "v", "vx", "vy", "vz"]
            varlabels = [r"$\rho$", r"$P$", r"$v$", r"$v_x$", r"$v_y$", r"$v_z$"]
            for i, j in product([0, 1], [0, 1, 2]):
                im = self.plot_2d_slice(axs[i, j], param=varnames[i + j], **kwargs)
                im_ax = fig.add_axes(
                    [
                        axs[i, j].get_position().x0 + axs[i, j].get_position().width,
                        axs[i, j].get_position().y0,
                        0.02,
                        axs[i, j].get_position().height,
                    ]
                )
                fig.colorbar(im, cax=im_ax, label=varlabels[i + j])
            return fig, axs
        raise NotImplementedError(
            "Plotting is only implemented for 1D and 2D simulations."
        )
