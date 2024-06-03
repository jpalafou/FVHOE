import numpy as np
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.config import conservative_names
from fvhoe.fv import (
    fv_average,
    conservative_interpolation,
    interpolate_cell_centers,
    interpolate_fv_averages,
    transverse_reconstruction,
)
from fvhoe.hydro import compute_conservatives, compute_primitives, hydro_dt
from fvhoe.named_array import NamedCupyArray, NamedNumpyArray
from fvhoe.ode import ODE
from fvhoe.riemann_solvers import advection_upwind, hllc, llf
from fvhoe.slope_limiting import (
    broadcase_troubled_cells_to_troubled_interfaces,
    detect_troubled_cells,
    MUSCL_interpolations,
)
from typing import Iterable, Tuple


class EulerSolver(ODE):
    def __init__(
        self,
        w0: callable,
        nx: int,
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
        riemann_solver: str = "HLLC",
        conservative_ic: bool = False,
        fixed_primitive_variables: Iterable = None,
        a_posteriori_slope_limiting: bool = False,
        slope_limiter: str = "minmod",
        progress_bar: bool = True,
        dumpall: bool = False,
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
                "HLLC" : advanced riemann solver for Euler equations
            conservative_ic (bool) : indicates that w0 returns conservative variables if true
            fixed_primitive_variables (Iterable) : series of primitive variables to keep fixed to their initial value
            a_posteriori_slope_limiting (bool) : whether to apply a postreiori slope limiting
            slope_limiter (str) : slope limiter code, "minmod", "moncen", None
            progress_bar (bool) : whether to print out a progress bar
            dumpall (bool) : save all variables in snapshot
            cupy (bool) : whether to use GPUs via the cupy library
        returns:
            EulerSolver object
        """

        # generate txyz mesh
        self.x_domain = x
        self.y_domain = y
        self.z_domain = z
        self.xi = np.linspace(x[0], x[1], nx + 1)  # x-interfaces
        self.yi = np.linspace(y[0], y[1], ny + 1)  # y-interfaces
        self.zi = np.linspace(z[0], z[1], nz + 1)  # z-interfaces
        self.x = 0.5 * (self.xi[:-1] + self.xi[1:])  # x-centers
        self.y = 0.5 * (self.yi[:-1] + self.yi[1:])  # y-centers
        self.z = 0.5 * (self.zi[:-1] + self.zi[1:])  # z-centers
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        hx = (x[1] - x[0]) / nx
        hy = (y[1] - y[0]) / ny
        hz = (z[1] - z[0]) / nz
        self.n = (nx, ny, nz)
        self.h = (hx, hy, hz)
        self.p = (px, py, pz)
        self.CFL = CFL
        self.fixed_dt = fixed_dt

        # physics
        self.gamma = gamma

        # riemann solver
        match riemann_solver:
            case "advection_upwind":
                self.riemann_solver = advection_upwind
            case "llf":
                self.riemann_solver = llf
            case "hllc":
                self.riemann_solver = hllc
            case _:
                raise TypeError(f"Invalid Riemann solver {riemann_solver}")

        # GPU
        self.cupy = cupy
        self.NamedArray = NamedCupyArray if cupy else NamedNumpyArray

        # boundary conditions
        bc = BoundaryCondition() if bc is None else bc
        self.bc = BoundaryCondition(
            names=conservative_names,
            x=bc.x,
            y=bc.x,
            z=bc.z,
            x_value=bc.x_value,
            y_value=bc.y_value,
            z_value=bc.z_value,
            x_domain=self.x_domain,
            y_domain=self.y_domain,
            z_domain=self.z_domain,
            h=self.h,
            p=self.p,
        )

        # integrator
        if conservative_ic:
            u0 = w0
        else:

            def u0(x, y, z):
                return compute_conservatives(w0(x, y, z), gamma=self.gamma)

        u0_fv = fv_average(f=u0, x=self.X, y=self.Y, z=self.Z, h=self.h, p=self.p)
        u0_fv = self.NamedArray(u0_fv, u0_fv.variable_names)
        super().__init__(u0_fv, progress_bar=progress_bar)
        self.dumpall = dumpall

        # fixed velocity
        self.fixed_primitive_variables = fixed_primitive_variables
        if fixed_primitive_variables is not None:
            self.u0 = u0_fv
            self.w0_cell_centers_cache = {}

        # slope limiting
        self.a_posteriori_slope_limiting = a_posteriori_slope_limiting
        self.slope_limiter = slope_limiter

    def f(self, t, u):
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
        dt, (F, G, H) = self.hydrofluxes(u=u, p=self.p)

        if self.a_posteriori_slope_limiting:
            self.revise_fluxes(u=u, F=F, G=G, H=H, dt=dt)

        # compute conservative variable dynamics
        dudt = self.euler_equation(F=F, G=G, H=H)
        return dt, dudt

    def hydrofluxes(
        self, u: NamedNumpyArray, p: Tuple[int, int, int], slope_limiter: str = None
    ) -> Tuple[float, Tuple[NamedNumpyArray, NamedNumpyArray, NamedNumpyArray]]:
        """
        compute the Euler equation fluxes F, G, H with degree p polynomials
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
        w_bc = self.interpolate_fvprimitives_from_fvconservatives(
            u=u,
            p=p,
            gw=(
                2 * int(np.ceil(p[0] / 2)) + gw,
                2 * int(np.ceil(p[1] / 2)) + gw,
                2 * int(np.ceil(p[2] / 2)) + gw,
            ),
        )

        # compute sound speed
        if np.min(w_bc.rho) < 0:
            raise BaseException("Negative density encountered.")
        if np.min(w_bc.P) < 0:
            raise BaseException("Negative pressure encountered.")

        # and time-step size
        if self.fixed_dt is None:
            dt = hydro_dt(w=w_bc, h=min(self.h), CFL=self.CFL, gamma=self.gamma)
        else:
            dt = self.fixed_dt

        # interpolate x face midpoints
        w_xy = conservative_interpolation(w_bc, p=p[2], axis=3, pos="c")
        w_x = conservative_interpolation(w_xy, p=p[1], axis=2, pos="c")
        if limit_slopes and p[0] == 1:
            w_x_face_center_l, w_x_face_center_r = MUSCL_interpolations(
                w_x, axis=1, limiter=slope_limiter
            )
        else:
            w_x_face_center_l = conservative_interpolation(w_x, p=p[0], axis=1, pos="l")
            w_x_face_center_r = conservative_interpolation(w_x, p=p[0], axis=1, pos="r")

        # interpolate y face midpoints
        w_yz = conservative_interpolation(w_bc, p=p[0], axis=1, pos="c")
        w_y = conservative_interpolation(w_yz, p=p[2], axis=3, pos="c")
        if limit_slopes and p[1] == 1:
            w_y_face_center_l, w_y_face_center_r = MUSCL_interpolations(
                w_y, axis=2, limiter=slope_limiter
            )
        else:
            w_y_face_center_l = conservative_interpolation(w_y, p=p[1], axis=2, pos="l")
            w_y_face_center_r = conservative_interpolation(w_y, p=p[1], axis=2, pos="r")

        # interpolate z face midpoints
        w_zx = conservative_interpolation(w_bc, p=p[1], axis=2, pos="c")
        w_z = conservative_interpolation(w_zx, p=p[0], axis=1, pos="c")
        if limit_slopes and p[2] == 1:
            w_z_face_center_l, w_z_face_center_r = MUSCL_interpolations(
                w_z, axis=3, limiter=slope_limiter
            )
        else:
            w_z_face_center_l = conservative_interpolation(w_z, p=p[2], axis=3, pos="l")
            w_z_face_center_r = conservative_interpolation(w_z, p=p[2], axis=3, pos="r")

        # pointwise numerical fluxes
        f_face_center = self.riemann_solver(
            wl=w_x_face_center_r[:, :-1, ...],
            wr=w_x_face_center_l[:, 1:, ...],
            gamma=self.gamma,
            dim="x",
        )
        g_face_center = self.riemann_solver(
            wl=w_y_face_center_r[:, :, :-1, ...],
            wr=w_y_face_center_l[:, :, 1:, ...],
            gamma=self.gamma,
            dim="y",
        )
        h_face_center = self.riemann_solver(
            wl=w_z_face_center_r[:, :, :, :-1, ...],
            wr=w_z_face_center_l[:, :, :, 1:, ...],
            gamma=self.gamma,
            dim="z",
        )

        # excess ghost zone counts after flux integral
        x_excess = (
            0,
            gw - int(np.ceil(p[0] / 2)) - 1,
            gw - 2 * int(np.ceil(p[1] / 2)),
            gw - 2 * int(np.ceil(p[2] / 2)),
        )
        y_excess = (
            0,
            gw - 2 * int(np.ceil(p[0] / 2)),
            gw - int(np.ceil(p[1] / 2)) - 1,
            gw - 2 * int(np.ceil(p[2] / 2)),
        )
        z_excess = (
            0,
            gw - 2 * int(np.ceil(p[0] / 2)),
            gw - 2 * int(np.ceil(p[1] / 2)),
            gw - int(np.ceil(p[2] / 2)) - 1,
        )

        # flux integrals
        F = transverse_reconstruction(
            transverse_reconstruction(f_face_center, axis=2, p=p[1]), axis=3, p=p[2]
        )[tuple(slice(x_excess[i] or None, -x_excess[i] or None) for i in range(4))]
        G = transverse_reconstruction(
            transverse_reconstruction(g_face_center, axis=1, p=p[0]), axis=3, p=p[2]
        )[tuple(slice(y_excess[i] or None, -y_excess[i] or None) for i in range(4))]
        H = transverse_reconstruction(
            transverse_reconstruction(h_face_center, axis=1, p=p[0]), axis=2, p=p[1]
        )[tuple(slice(z_excess[i] or None, -z_excess[i] or None) for i in range(4))]

        return dt, (F, G, H)

    def revise_fluxes(
        self,
        u: NamedNumpyArray,
        F: NamedNumpyArray,
        G: NamedNumpyArray,
        H: NamedNumpyArray,
        dt: float,
    ) -> None:
        """
        revise fluxes to prevent oscillations
        args:
            u (NamedNumpyArray) : fv averages of conservative variables. has shape (5, nx, ny, nz)
            F (NamedNumpyArray) : x-fluxes. has shape (5, nx + 1, ny, nz)
            G (NamedNumpyArray) : y-fluxes. has shape (5, nx, ny + 1, nz)
            H (NamedNumpyArray) : z-fluxes. has shape (5, nx, ny, nz + 1)
            dt (float) : time-step size
        returns:
            None : revise fluxes in place
        """
        # compute candidate solution
        ustar = u + dt * self.euler_equation(F=F, G=G, H=H)
        ustar_bc = self.bc.apply(ustar, gw=(1, 1, 1))
        u_bc = self.bc.apply(u, gw=(1, 1, 1))

        # detect troubled cells
        troubled_cells = detect_troubled_cells(
            u=u_bc,
            u_candidate=ustar_bc,
            eps=1e-5,
            xp={True: "cupy", False: "numpy"}[self.cupy],
        )

        if not np.any(troubled_cells):
            # great, no troubled cells!
            return

        # p=1, slope-limited fluxes
        _, (Fl, Gl, Hl) = self.hydrofluxes(
            u=u,
            p=(int(self.p[0] > 0), int(self.p[1] > 0), int(self.p[2] > 0)),
            slope_limiter=self.slope_limiter,
        )
        troubled_x, troubled_y, troubled_z = (
            broadcase_troubled_cells_to_troubled_interfaces(
                troubled_cells, xp={True: "cupy", False: "numpy"}[self.cupy]
            )
        )
        F[...] = np.where(troubled_x, Fl, F)
        G[...] = np.where(troubled_y, Gl, G)
        H[...] = np.where(troubled_z, Hl, H)

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
        dudt = -(1 / self.h[0]) * (F[:, 1:, ...] - F[:, :-1, ...])
        dudt += -(1 / self.h[1]) * (G[:, :, 1:, ...] - G[:, :, :-1, ...])
        dudt += -(1 / self.h[2]) * (H[:, :, :, 1:, ...] - H[:, :, :, :-1, ...])
        return dudt

    def interpolate_fvprimitives_from_fvconservatives(
        self,
        u: NamedNumpyArray,
        p: Tuple[int, int, int],
        gw: Tuple[int, int, int],
    ) -> NamedNumpyArray:
        """
        interpolate finite volume primitives from finite volume conservatives
        args:
            u (NamedArray) : finite volume conservative variables
            p (Tuple[int, int, int]) : interpolation polynomial degree in 3D (px, py, pz)
            gw (Tuple[int, int, int]) : number of 'ghost zones' on either side of the array u in each direction (gwx, gwy, gwz)
            gamma (float) : specific heat ratio
        returns:
            w (NamedArray) : finite volume primitive variables
        """
        u_bc = self.bc.apply(u, gw=gw)
        u_cell_centers = interpolate_cell_centers(u_bc, p=p)
        w_cell_centers = compute_primitives(u_cell_centers, gamma=self.gamma)
        if self.fixed_primitive_variables is not None and self.t > 0:
            w0_cell_centers = self.w0_cell_centers_cache.get(gw, None)
            if w0_cell_centers is None:
                u0_bc = self.bc.apply(self.u0, gw=gw)
                u0_cell_centers = interpolate_cell_centers(u0_bc, p=p)
                w0_cell_centers = compute_primitives(u0_cell_centers, gamma=self.gamma)
                self.w0_cell_centers_cache[gw] = w0_cell_centers
            for var in self.fixed_primitive_variables:
                setattr(w_cell_centers, var, getattr(w0_cell_centers, var))
        w = interpolate_fv_averages(w_cell_centers, p=p)
        return w

    def snapshot(self):
        """
        log a dictionary
        """
        # fv conservatives to fv primitives
        w = self.interpolate_fvprimitives_from_fvconservatives(
            u=self.u,
            p=self.p,
            gw=(
                2 * int(np.ceil(self.p[0] / 2)),
                2 * int(np.ceil(self.p[1] / 2)),
                2 * int(np.ceil(self.p[2] / 2)),
            ),
        )

        # convert to numpy
        if self.cupy:
            w = w.asnamednumpy()

        if self.dumpall:
            u = self.u.asnamednumpy() if self.cupy else self.u.copy()
            fv_data = u.merge(w)

        log = {
            "t": self.t,
            "fv": fv_data if self.dumpall else w,
        }
        self.snapshots.append(log)
        self.snapshot_times.append(self.t)

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

    def plot_1d_slice(
        self,
        ax,
        param: str,
        t: float = None,
        x=None,
        y=None,
        z=None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        plot a 1-dimensional slice by specifying t and two of three spatial dimensions x, y, and z
        """
        if sum([x is None, y is None, z is None]) != 1:
            raise BaseException("One out of the three coordinates x-y-z must be None")
        t = max(self.snapshot_times) if t is None else t
        n = np.argmin(np.abs(np.array(list(self.snapshot_times)) - t))
        t = list(self.snapshot_times)[n]
        if x is None:
            j, k = np.argmin(np.abs(self.y - y)), np.argmin(np.abs(self.z - z))
            y, z = self.y[j], self.z[k]
            x = self.x
            x_for_plotting = self.x
            y_for_plotting = getattr(self.snapshots[n]["fv"], param)[:, j, k]
        elif y is None:
            i, k = np.argmin(np.abs(self.x - x)), np.argmin(np.abs(self.z - z))
            x, z = self.x[i], self.z[k]
            y = self.y
            x_for_plotting = self.y
            y_for_plotting = getattr(self.snapshots[n]["fv"], param)[i, :, k]
        elif z is None:
            i, j = np.argmin(np.abs(self.x - x)), np.argmin(np.abs(self.y - y))
            x, y = self.x[i], self.y[j]
            z = self.z
            x_for_plotting = self.z
            y_for_plotting = getattr(self.snapshots[n]["fv"], param)[i, j, :]
        if verbose:
            t_message = f"{t:.2f}"
            x_message = (
                f"{x:.2f}"
                if (isinstance(x, int) or isinstance(x, float))
                else f"[{x[0]:.2f}, {x[-1]:.2f}]"
            )
            y_message = (
                f"{y:.2f}"
                if (isinstance(y, int) or isinstance(y, float))
                else f"[{y[0]:.2f}, {y[-1]:.2f}]"
            )
            z_message = (
                f"{z:.2f}"
                if (isinstance(z, int) or isinstance(z, float))
                else f"[{z[0]:.2f}, {z[-1]:.2f}]"
            )
            print(f"t={t_message}, x={x_message}, y={y_message}, z={z_message}")
        return ax.plot(x_for_plotting, y_for_plotting, **kwargs)

    def plot_2d_slice(
        self,
        ax,
        param: str,
        t: float = None,
        x=None,
        y=None,
        z=None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        plot a 2-dimensional slice by specifying t and one of three spatial dimensions x, y, and z
        """
        if sum([x is None, y is None, z is None]) != 2:
            raise BaseException("Two out of the three coordinates x-y-z must be None")
        t = max(self.snapshot_times) if t is None else t
        n = np.argmin(np.abs(np.array(list(self.snapshot_times)) - t))
        t = list(self.snapshot_times)[n]
        if x is None and y is None:
            k = np.argmin(np.abs(self.z - z))
            z = self.z[k]
            x, y = self.x, self.y
            z_for_plotting = getattr(self.snapshots[n]["fv"], param)[:, :, k]
            z_for_plotting = np.rot90(z_for_plotting, 1)
            horizontal_axis, vertical_axis = "x", "y"
            limits = (x[0], x[-1], y[0], y[-1])
        elif y is None and z is None:
            i = np.argmin(np.abs(self.x - x))
            x = self.x[i]
            y, z = self.y, self.z
            z_for_plotting = getattr(self.snapshots[n]["fv"], param)[i, :, :]
            z_for_plotting = np.rot90(z_for_plotting, 1)
            horizontal_axis, vertical_axis = "y", "z"
            limits = (y[0], y[-1], z[0], z[-1])
        elif x is None and z is None:
            j = np.argmin(np.abs(self.y - y))
            y = self.y[j]
            z, x = self.z, self.x
            z_for_plotting = getattr(self.snapshots[n]["fv"], param)[:, j, :]
            z_for_plotting = np.rot90(z_for_plotting, 1)
            horizontal_axis, vertical_axis = "x", "z"
            limits = (x[0], x[-1], z[0], z[-1])
        if verbose:
            t_message = f"{t:.2f}"
            x_message = (
                f"{x:.2f}"
                if (isinstance(x, int) or isinstance(x, float))
                else f"[{x[0]:.2f}, {x[-1]:.2f}]"
            )
            y_message = (
                f"{y:.2f}"
                if (isinstance(y, int) or isinstance(y, float))
                else f"[{y[0]:.2f}, {y[-1]:.2f}]"
            )
            z_message = (
                f"{z:.2f}"
                if (isinstance(z, int) or isinstance(z, float))
                else f"[{z[0]:.2f}, {z[-1]:.2f}]"
            )
            print(f"t={t_message}, x={x_message}, y={y_message}, z={z_message}")
            print(f"{horizontal_axis=}, {vertical_axis=}")
        return ax.imshow(z_for_plotting, extent=limits, **kwargs)
