from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.config import conservative_names, primitive_names
from fvhoe.fv import (
    conservative_interpolation,
    fv_average,
    fv_uniform_meshgen,
    interpolate_cell_centers,
    interpolate_fv_averages,
    transverse_reconstruction,
)
from fvhoe.hydro import compute_conservatives, compute_primitives, hydro_dt
from fvhoe.initial_conditions import square
from fvhoe.named_array import NamedCupyArray, NamedNumpyArray
from fvhoe.ode import ODE
from fvhoe.riemann_solvers import advection_upwind, hllc, llf
from fvhoe.slope_limiting import (
    broadcast_to_troubled_interfaces,
    detect_troubled_cells,
    MUSCL_interpolations,
)
from fvhoe.timer import Timer
from fvhoe.visualization import plot_1d_slice, plot_2d_slice
from itertools import product
import json
import numpy as np
import os
import shutil
import pickle
from typing import Iterable, Tuple

try:
    import cupy as cp
    from cupy import ndarray as cp_ndarray

    CUPY_AVAILABLE = True
except Exception:
    from numpy import ndarray as cp_ndarray

    CUPY_AVAILABLE = False


class EulerSolver(ODE):
    def __init__(
        self,
        w0: callable = square,
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
        NAD_mode: str = "any",
        NAD_vars: list = None,
        PAD: dict = None,
        SED: bool = True,
        SED_tolerance: float = 1e-10,
        convex: bool = False,
        density_floor: bool = False,
        pressure_floor: bool = False,
        rho_P_sound_speed_floor: bool = False,
        all_floors: bool = False,
        progress_bar: bool = True,
        dumpall: bool = False,
        snapshots_as_fv_averages: bool = True,
        snapshot_helper_function: callable = None,
        slab_buffer_size: int = 30,
        cupy: bool = False,
    ):
        """
        solver for Euler equations, a system of 5 variables:
            rho (density)
            P (pressure)
            vx (x-velocity)
            vy (y-velocity)
            vz (z-velocity)
        implemented in 1D, 2D, and 3D
        args:
            w0(X, Y, Z) (callable) : function of a 3D mesh. returns NamedArray of shape
                (5, nz, ny, nz)
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
            NAD_mode (str) : NAD mode in troubled cell detection: "any", "only"
            NAD_vars (list) : when NAD_mode is "only", list of variables to apply NAD
            PAD (dict) : primitive variable limits for slope limiting
            SED (bool) : whether to ignore NAD trouble where smooth extrema are detected
            SED_tolerance (float) : tolerance for avoiding dividing by 0 in smooth extrema detection
            convex (bool) : whether to apply convex slope limiting
            density_floor (bool) : whether to apply a density floor
            pressure_floor (bool) : whether to apply a pressure floor
            rho_P_sound_speed_floor (bool) : whether to apply a pressure and density floor in the sound speed function
            all_floors (bool) : apply all floors
            progress_bar (bool) : whether to print out a progress bar
            dumpall (bool) : save all variables in snapshot
            snapshots_as_fv_averages (bool) : save snapshots as finite volume averages. if false, save as cell centers
            snapshot_helper_function (callable) : function to call at the end of a snapshot with self as the sole argument
            slab_buffer_size (int) : for applying boundary conditions
            cupy (bool) : whether to use GPUs via the cupy library
        returns:
            EulerSolver object
        """

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
        self.CFL = CFL
        self.fixed_dt = fixed_dt

        # physics
        self.gamma = gamma

        # riemann solver
        self.riemann_solver_name = riemann_solver
        match riemann_solver:
            case "advection_upwind":
                self.riemann_solver = advection_upwind
            case "llf":
                self.riemann_solver = llf
            case "hllc":
                self.riemann_solver = hllc
            case _:
                raise TypeError(f"Invalid Riemann solver {riemann_solver}")

        # data management
        self.dumpall = dumpall
        self.snapshots_as_fv_averages = snapshots_as_fv_averages
        self.snapshot_helper_function = snapshot_helper_function
        self.cupy = cupy
        self.NamedArray = NamedCupyArray if cupy else NamedNumpyArray
        self.timeseries_E = np.array([])
        self.timeseries_rho = np.array([])

        # timing
        self.timer = Timer(
            [
                "bc",
                "hydrodynamics",
                "hydrodynamics/hydrofluxes",
                "revise_fluxes",
                "revise_fluxes/broadcast_to_troubled_interfaces",
                "revise_fluxes/detect_troubled_cells",
                "revise_fluxes/hydrofluxes",
                "revise_fluxes/MUSCL_interpolations",
                "riemann_solver",
                "snapshot",
            ]
        )

        # initial conditions
        self.w0 = w0
        if conservative_ic:
            u0 = w0
        else:

            def u0(x, y, z):
                return compute_conservatives(w0(x, y, z), gamma=self.gamma)

        if fv_ic:
            u0_fv = u0(x=self.X, y=self.Y, z=self.Z)
        else:
            u0_fv = fv_average(f=u0, x=self.X, y=self.Y, z=self.Z, h=self.h, p=self.p)
        u0_fv = self.NamedArray(u0_fv, u0_fv.variable_names)

        # get slab coordinates for boundaries
        slab_buffer_sizes = (
            slab_buffer_size if self.xdim else 0,
            slab_buffer_size if self.ydim else 0,
            slab_buffer_size if self.zdim else 0,
        )
        _, slab_coords = fv_uniform_meshgen(
            (nx, ny, nz), x=x, y=y, z=z, slab_thickness=slab_buffer_sizes
        )

        # define boundary conditions
        self.bc = (
            BoundaryCondition(
                x=("periodic", "periodic"),
                y=("periodic", "periodic"),
                z=("periodic", "periodic"),
            )
            if bc is None
            else bc
        )
        self.bc.slab_coords = slab_coords

        # configure initial condition boundaries
        any_ic_bc = False
        for dim, i in product(["x", "y", "z"], (0, 1)):
            if getattr(self.bc, dim)[i] == "ic":
                any_ic_bc = True
                dim_value = list(getattr(self.bc, f"{dim}_value"))
                dim_value[i] = u0
                setattr(self.bc, f"{dim}_value", dim_value)
        if any_ic_bc and fv_ic:
            print(
                "Warning: initial condition function returns finite volume averages and is being used to apply boundary conditions."
            )
        self.bc.__post_init__()

        # inegrator class init
        super().__init__(u0_fv, progress_bar=progress_bar)

        # fixed velocity
        self.fixed_primitive_variables = fixed_primitive_variables
        if fixed_primitive_variables is not None:
            self.u0 = u0_fv
            self.w0_cell_centers_cache = {}

        # slope limiting
        self.a_posteriori_slope_limiting = a_posteriori_slope_limiting
        self.slope_limiter = slope_limiter
        self.force_trouble = force_trouble
        self.NAD = NAD
        self.NAD_mode = NAD_mode
        self.NAD_vars = NAD_vars
        self.PAD = PAD if isinstance(PAD, dict) else {}
        defaults_limits = {
            "rho": (0.0, np.inf),
            "P": (0.0, np.inf),
            "vx": (-np.inf, np.inf),
            "vy": (-np.inf, np.inf),
            "vz": (-np.inf, np.inf),
        }
        for var in primitive_names:
            if var not in self.PAD.keys():
                self.PAD[var] = defaults_limits[var]
        self.SED = SED
        self.SED_tolerance = SED_tolerance
        self.convex = convex
        self.trouble = np.zeros_like(u0_fv[0])
        self.NAD_violation_magnitude = np.zeros_like(u0_fv[0])
        self.trouble_counter = 1

        # floors
        self.density_floor = density_floor or all_floors
        self.pressure_floor = pressure_floor or all_floors
        self.rho_P_sound_speed_floor = rho_P_sound_speed_floor or all_floors

        # plotting functions
        self.plot_1d_slice = lambda *args, **kwargs: plot_1d_slice(
            self, *args, **kwargs
        )
        self.plot_2d_slice = lambda *args, **kwargs: plot_2d_slice(
            self, *args, **kwargs
        )

        # misc
        self.f_evaluation_count = 0

        # preallocate flux arrays
        self.F = self.NamedArray(np.empty((5, nx + 1, ny, nz)), conservative_names)
        self.G = self.NamedArray(np.empty((5, nx, ny + 1, nz)), conservative_names)
        self.H = self.NamedArray(np.empty((5, nx, ny, nz + 1)), conservative_names)

    def f(self, t, u):
        self.f_evaluation_count += 1
        return self.hydrodynamics(u)

    def hydrodynamics(self, u: NamedNumpyArray) -> Tuple[float, NamedNumpyArray]:
        """
        compute the Euler equation dynamics from cell-average conserved values u
        args:
            u (NamedNumpyArray) : fv averages of conservative variables. has shape (5, nx, ny, nz)
        returns:
            dt, dudt (float, NamedNumpyArray) : time-step size, conservative variable dynamics
        """
        # high-order fluxes
        self.timer.start("hydrodynamics")
        self.timer.start("hydrodynamics/hydrofluxes")
        dt, (self.F[...], self.G[...], self.H[...]) = self.hydrofluxes(u=u, p=self.p)
        self.timer.stop("hydrodynamics/hydrofluxes")

        if self.a_posteriori_slope_limiting:
            self.timer.start("revise_fluxes")
            self.revise_fluxes(
                u=u,
                F=self.F,
                G=self.G,
                H=self.H,
                dt=dt,
                force_trouble=self.force_trouble,
            )
            self.timer.stop("revise_fluxes")

        # compute conservative variable dynamics
        dudt = self.euler_equation(F=self.F, G=self.G, H=self.H)
        self.timer.stop("hydrodynamics")
        return dt, dudt

    def hydrofluxes(
        self, u: NamedNumpyArray, p: Tuple[int, int, int], slope_limiter: str = None
    ) -> Tuple[float, Tuple[NamedNumpyArray, NamedNumpyArray, NamedNumpyArray]]:
        """
        compute the Euler equation fluxes F, G, H with degree p polynomials. optionally apply slope limiting.
        if xdim is False, F is returned as self.F and so on for G and H
        args:
            u (NamedArray) : cell-averages of conservative variables. has shape (5, nx, ny, nz)
            p (Tuple[int, int, int]) : polynomial interpolation degree (px, py, pz)
            limiter (str) : None, 'minmod', 'moncen'
        returns:
            dt, (F, G, H) (tuple) : time-step size and numerical fluxes
                dt (float) : time-step size
                F (NamedArray) : x-direction conservative fluxes. has shape (5, nx + 1, ny, nz)
                G (NamedArray) : y-direction conservative fluxes. has shape (5, nx, ny + 1, nz)
                H (NamedArray) : z-direction conservative fluxes. has shape (5, nx, ny, nz + 1)
        """
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
                2 * int(np.ceil(p[0] / 2)) + gw if self.xdim else 0,
                2 * int(np.ceil(p[1] / 2)) + gw if self.ydim else 0,
                2 * int(np.ceil(p[2] / 2)) + gw if self.zdim else 0,
            ),
        )

        # check solution for invalid values
        if self.density_floor:
            w_bc.rho = np.maximum(w_bc.rho, 1e-16)
        elif np.min(w_bc.rho) < 0:
            raise BaseException("Negative density encountered.")

        if self.pressure_floor:
            w_bc.P = np.maximum(w_bc.P, 1e-16)
        elif np.min(w_bc.P) < 0:
            raise BaseException("Negative pressure encountered.")

        if np.any(np.isnan(w_bc.rho)):
            raise BaseException("NaNs encountered in density.")

        if np.any(np.isnan(w_bc.P)):
            raise BaseException("NaNs encountered in pressure.")

        # and time-step size
        if self.fixed_dt is None:
            dt = hydro_dt(
                w=w_bc,
                h=min(self.h),
                ndim=self.ndim,
                CFL=self.CFL,
                gamma=self.gamma,
                rho_P_sound_speed_floor=self.rho_P_sound_speed_floor,
            )
        else:
            dt = self.fixed_dt

        # interpolate x face midpoints
        if self.xdim:
            w_xy = conservative_interpolation(w_bc, p=p[2], axis=3, pos="c")
            w_x = conservative_interpolation(w_xy, p=p[1], axis=2, pos="c")
            if limit_slopes and p[0] == 1:
                self.timer.start("revise_fluxes/MUSCL_interpolations")
                w_x_face_center_l, w_x_face_center_r = MUSCL_interpolations(
                    w_x, axis=1, limiter=slope_limiter
                )
                self.timer.stop("revise_fluxes/MUSCL_interpolations")
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
                self.timer.start("revise_fluxes/MUSCL_interpolations")
                w_y_face_center_l, w_y_face_center_r = MUSCL_interpolations(
                    w_y, axis=2, limiter=slope_limiter
                )
                self.timer.stop("revise_fluxes/MUSCL_interpolations")
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
                self.timer.start("revise_fluxes/MUSCL_interpolations")
                w_z_face_center_l, w_z_face_center_r = MUSCL_interpolations(
                    w_z, axis=3, limiter=slope_limiter
                )
                self.timer.stop("revise_fluxes/MUSCL_interpolations")
            else:
                w_z_face_center_l = conservative_interpolation(
                    w_z, p=p[2], axis=3, pos="l"
                )
                w_z_face_center_r = conservative_interpolation(
                    w_z, p=p[2], axis=3, pos="r"
                )

        # pointwise numerical fluxes
        self.timer.start("riemann_solver")
        if self.xdim:
            f_face_center = self.riemann_solver(
                wl=w_x_face_center_r[:, :-1, ...],
                wr=w_x_face_center_l[:, 1:, ...],
                gamma=self.gamma,
                dim="x",
                rho_P_sound_speed_floor=self.rho_P_sound_speed_floor,
            )
        if self.ydim:
            g_face_center = self.riemann_solver(
                wl=w_y_face_center_r[:, :, :-1, ...],
                wr=w_y_face_center_l[:, :, 1:, ...],
                gamma=self.gamma,
                dim="y",
                rho_P_sound_speed_floor=self.rho_P_sound_speed_floor,
            )
        if self.zdim:
            h_face_center = self.riemann_solver(
                wl=w_z_face_center_r[:, :, :, :-1, ...],
                wr=w_z_face_center_l[:, :, :, 1:, ...],
                gamma=self.gamma,
                dim="z",
                rho_P_sound_speed_floor=self.rho_P_sound_speed_floor,
            )
        self.timer.stop("riemann_solver")

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
        if self.xdim:
            F = transverse_reconstruction(
                transverse_reconstruction(f_face_center, axis=2, p=p[1]), axis=3, p=p[2]
            )[tuple(slice(x_excess[i] or None, -x_excess[i] or None) for i in range(4))]
        else:
            F = self.F
        if self.ydim:
            G = transverse_reconstruction(
                transverse_reconstruction(g_face_center, axis=1, p=p[0]), axis=3, p=p[2]
            )[tuple(slice(y_excess[i] or None, -y_excess[i] or None) for i in range(4))]
        else:
            G = self.G
        if self.zdim:
            H = transverse_reconstruction(
                transverse_reconstruction(h_face_center, axis=1, p=p[0]), axis=2, p=p[1]
            )[tuple(slice(z_excess[i] or None, -z_excess[i] or None) for i in range(4))]
        else:
            H = self.H
        return dt, (F, G, H)

    def revise_fluxes(
        self,
        u: NamedNumpyArray,
        F: NamedNumpyArray,
        G: NamedNumpyArray,
        H: NamedNumpyArray,
        dt: float,
        force_trouble: bool = False,
    ) -> None:
        """
        revise fluxes to prevent oscillations
        args:
            u (NamedNumpyArray) : fv averages of conservative variables. has shape (5, nx, ny, nz)
            F (NamedNumpyArray) : x-fluxes. has shape (5, nx + 1, ny, nz)
            G (NamedNumpyArray) : y-fluxes. has shape (5, nx, ny + 1, nz)
            H (NamedNumpyArray) : z-fluxes. has shape (5, nx, ny, nz + 1)
            dt (float) : time-step size
            force_trouble (bool) : all cells are troubled
        returns:
            None : revise fluxes in place
        """
        # compute candidate solution
        ustar = u + dt * self.euler_equation(F=F, G=G, H=H)

        # interpolate finite volume primitives from finite volume conservatives
        gws = (
            2 * int(np.ceil(self.p[0] / 2)) + 3 if self.xdim else 0,
            2 * int(np.ceil(self.p[1] / 2)) + 3 if self.ydim else 0,
            2 * int(np.ceil(self.p[2] / 2)) + 3 if self.zdim else 0,
        )
        w = self.interpolate_primitives_from_conservatives(u, p=self.p, gw=gws)
        w_star = self.interpolate_primitives_from_conservatives(ustar, p=self.p, gw=gws)

        # detect troubled cells
        self.timer.start("revise_fluxes/detect_troubled_cells")
        troubled_cells, NAD_mag = detect_troubled_cells(
            u=w,
            u_candidate=w_star,
            dims=self.dims,
            NAD_tolerance=self.NAD,
            NAD_mode=self.NAD_mode,
            NAD_vars=self.NAD_vars,
            PAD=self.PAD,
            SED=self.SED,
            SED_tolerance=self.SED_tolerance,
            xp={True: "cupy", False: "numpy"}[self.cupy],
        )
        self.timer.stop("revise_fluxes/detect_troubled_cells")
        if force_trouble:
            troubled_cells = np.ones_like(troubled_cells, dtype=bool)

        self.log_troubles(troubled_cells, NAD_mag)

        if not np.any(troubled_cells):
            # great, no troubled cells!
            return

        # p=1, slope-limited fluxes
        self.timer.start("revise_fluxes/hydrofluxes")
        _, (Fl, Gl, Hl) = self.hydrofluxes(
            u=u,
            p=(int(self.p[0] > 0), int(self.p[1] > 0), int(self.p[2] > 0)),
            slope_limiter=self.slope_limiter,
        )
        self.timer.stop("revise_fluxes/hydrofluxes")
        self.timer.start("revise_fluxes/broadcast_to_troubled_interfaces")
        troubled_x, troubled_y, troubled_z = broadcast_to_troubled_interfaces(
            troubled_cells,
            dims=self.dims,
            convex=self.convex,
            periodic_x=self.bc.x == ("periodic", "periodic"),
            periodic_y=self.bc.y == ("periodic", "periodic"),
            periodic_z=self.bc.z == ("periodic", "periodic"),
            xp={True: "cupy", False: "numpy"}[self.cupy],
        )
        self.timer.stop("revise_fluxes/broadcast_to_troubled_interfaces")

        F[...] = (1 - troubled_x) * F + troubled_x * Fl
        G[...] = (1 - troubled_y) * G + troubled_y * Gl
        H[...] = (1 - troubled_z) * H + troubled_z * Hl

    def euler_equation(
        self, F: NamedNumpyArray, G: NamedNumpyArray, H: NamedNumpyArray
    ) -> NamedNumpyArray:
        """
        compute the Euler equation dynamics from fv-average conserved values u
        args:
            F (NamedNumpyArray) : x-fluxes. has shape (5, nx + 1, ny, nz)
            G (NamedNumpyArray) : y-fluxes. has shape (5, nx, ny + 1, nz)
            H (NamedNumpyArray) : z-fluxes. has shape (5, nx, ny, nz + 1)
        returns:
            dudt (NamedNumpyArray) : conservative variable dynamics
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
        u: NamedNumpyArray,
        p: Tuple[int, int, int],
        gw: Tuple[int, int, int],
        fv_average: bool = True,
    ) -> NamedNumpyArray:
        """
        interpolate finite volume primitives from finite volume conservatives
        args:
            u (NamedArray) : finite volume conservative variables
            p (Tuple[int, int, int]) : interpolation polynomial degree in 3D (px, py, pz)
            gw (Tuple[int, int, int]) : number of 'ghost zones' on either side of the array u in each direction (gwx, gwy, gwz)
            fv_average (bool) : whether u is cell-averaged or centroids
        returns:
            w (NamedArray) : primitive variables as finite volume averages or centroids
        """
        self.timer.start("bc")
        u_bc = self.bc.apply(u, gw=gw, t=self.t)
        self.timer.stop("bc")
        u_cell_centers = interpolate_cell_centers(u_bc, p=p)
        w_cell_centers = compute_primitives(u_cell_centers, gamma=self.gamma)
        if self.fixed_primitive_variables is not None and self.t > 0:
            w0_cell_centers = self.w0_cell_centers_cache.get(f"{gw=}, {p=}", None)
            if w0_cell_centers is None:
                self.timer.start("bc")
                u0_bc = self.bc.apply(self.u0, gw=gw, t=0)
                self.timer.stop("bc")
                u0_cell_centers = interpolate_cell_centers(u0_bc, p=p)
                w0_cell_centers = compute_primitives(u0_cell_centers, gamma=self.gamma)
                self.w0_cell_centers_cache[f"{gw=}, {p=}"] = w0_cell_centers
            for var in self.fixed_primitive_variables:
                setattr(w_cell_centers, var, getattr(w0_cell_centers, var))
        if fv_average:
            w = interpolate_fv_averages(w_cell_centers, p=p)
        else:
            w = w_cell_centers
        return w

    def log_troubles(
        self,
        trouble: np.ndarray,
        NAD_violation_magnitude: np.ndarray,
        reset: bool = False,
    ) -> None:
        """
        log troubled cells and PAD violation magnitude as the mean along each Runge-Kutta substep
        args:
            trouble (array_like) : array of troubled cells
            NAD_violation_magnitude (array_like) : array of PAD violation magnitudes
            reset (bool) : reset the trouble counter
        """
        if reset:
            self.trouble /= self.trouble_counter
            self.NAD_violation_magnitude /= self.trouble_counter
            self.trouble_counter = 0
            return
        if self.trouble_counter == 0:
            self.trouble[...] = trouble
            self.NAD_violation_magnitude[...] = NAD_violation_magnitude
        else:
            self.trouble += trouble
            self.NAD_violation_magnitude += NAD_violation_magnitude
        self.trouble_counter += 1

    def step_helper_function(self):
        # log timeseries data
        self.timeseries_E = np.append(self.timeseries_E, np.mean(self.u.E).item())
        self.timeseries_rho = np.append(self.timeseries_rho, np.mean(self.u.rho).item())
        # log
        if self.a_posteriori_slope_limiting:
            self.log_troubles(trouble=None, NAD_violation_magnitude=None, reset=True)

    def snapshot(self):
        """
        log a dictionary
        """
        self.timer.start("snapshot")
        # fv conservatives to fv primitives
        w = self.interpolate_primitives_from_conservatives(
            u=self.u,
            p=self.p,
            gw=(
                int(np.ceil(self.p[0] / 2)) * int(self.snapshots_as_fv_averages + 1),
                int(np.ceil(self.p[1] / 2)) * int(self.snapshots_as_fv_averages + 1),
                int(np.ceil(self.p[2] / 2)) * int(self.snapshots_as_fv_averages + 1),
            ),
            fv_average=self.snapshots_as_fv_averages,
        )

        # append conservative variables if dumpall
        if self.dumpall:
            if self.snapshots_as_fv_averages:
                u = self.u
            else:
                self.timer.start("bc")
                u_bc = self.bc.apply(
                    self.u,
                    gw=(
                        int(np.ceil(self.p[0] / 2)),
                        int(np.ceil(self.p[1] / 2)),
                        int(np.ceil(self.p[2] / 2)),
                    ),
                    t=self.t,
                )
                self.timer.stop("bc")
                u = interpolate_cell_centers(u_bc, p=self.p)
            w = w.merge(u)

        # log troubles
        if self.a_posteriori_slope_limiting:
            trouble = self.trouble
            NAD_mag = self.NAD_violation_magnitude

        # convert cupy arrays to numpy arrays
        if self.cupy and CUPY_AVAILABLE:
            w = w.asnamednumpy()
            if self.a_posteriori_slope_limiting:
                trouble = cp.asnumpy(self.trouble)
                NAD_mag = cp.asnumpy(self.NAD_violation_magnitude)

        # append dictionary to list of snapshots
        log = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "t": self.t,
            "w": w,
        }
        if self.a_posteriori_slope_limiting:
            log["trouble"] = trouble
            log["NAD violation magnitude"] = NAD_mag

        self.snapshots.append(log)
        self.snapshot_times.append(self.t)

        if self.snapshot_helper_function is not None:
            self.snapshot_helper_function(self)

        self.timer.stop("snapshot")

    def read_snapshots(self, overwrite: bool) -> bool:
        """
        read snapshots from a pickle file
        args:
            overwrite (bool) : overwrite the file if it exists
        returns:
            out (bool) : whether a file was found
        """
        if os.path.exists(self.snapshot_dir) and not overwrite:
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
                callable(v)
                or isinstance(v, np.ndarray)
                or isinstance(v, cp_ndarray)
                or isinstance(v, NamedNumpyArray)
                or isinstance(v, NamedCupyArray)
                or k
                in [
                    "NamedArray",
                    "progress_bar",
                    "timestamps",
                    "snapshots",
                    "w0_cell_centers_cache",
                ]
            ):
                continue
            # save boundary condition dict
            if k in ["bc", "timer"]:
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

        print(f"Wrote to snapshot directory {snapshot_dir}")

    def rkorder(self, *args, **kwargs):
        """
        chose Runge-Kutta method to match the spatial interpolation polynomial degree
        """
        p = max(self.p)
        match p:
            case 0:
                self.euler(*args, **kwargs)
            case 1:
                self.ssprk2(*args, **kwargs)
            case 2:
                self.ssprk3(*args, **kwargs)
            case _:
                self.rk4(*args, **kwargs)
