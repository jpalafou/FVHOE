import numpy as np
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.fv import fv_average, conservative_interpolation, transverse_reconstruction
from fvhoe.hydro import (
    compute_conservatives,
    compute_primitives,
    compute_sound_speed,
)
from fvhoe.named_array import NamedCupyArray, NamedNumpyArray
from fvhoe.ode import ODE
from fvhoe.riemann_solvers import advection_upwind, HLLC
from typing import Tuple


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
        progress_bar: bool = True,
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
            progress_bar (bool) : whether to print out a progress bar
            cupy (bool) : whether to use GPUs via the cupy library
        returns:
            EulerSolver object
        """

        # generate txyz mesh
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
        if riemann_solver == "HLLC":
            self.riemann_solver = HLLC
        elif riemann_solver == "advection_upwind":
            self.riemann_solver = advection_upwind
        else:
            raise TypeError(f"Invalid Riemann solver {riemann_solver}")

        # GPU
        self.cupy = cupy
        self.NamedArray = NamedCupyArray if cupy else NamedNumpyArray

        # boundary conditions
        self.bc = (
            bc if bc is not None else BoundaryCondition(NamedArray=self.NamedArray)
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

    def f(self, t, u):
        return self.hydrodynamics(u)

    def snapshot(self):
        """
        log a dictionary
        """

        u = self.u.asnamednumpy() if self.cupy else self.u.copy()
        w = compute_primitives(u, gamma=self.gamma)
        fv_data = u.merge(w)

        log = {
            "t": self.t,
            "fv": fv_data,
        }
        self.snapshots.append(log)
        self.snapshot_times.append(self.t)

    def interpolate_cell_centers(
        self, fvarr: np.ndarray, p: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        args:
            fvarr (array_like) : array of cell averages, shape (5, ny, nx,n z)
            p (Tuple[int, int, int]) : polynomial interpolation degree (px, py, pz)
        returns
            out (array_like) : conservative interpolation of fvarr at cell centers
        """
        gw = (
            int(np.ceil(self.p[0] / 2)),
            int(np.ceil(self.p[1] / 2)),
            int(np.ceil(self.p[2] / 2)),
        )
        fvarr_bced = self.bc.apply(fvarr, gw=(0, gw[0], gw[1], gw[2]))

        out = conservative_interpolation(
            fvarr=conservative_interpolation(
                fvarr=conservative_interpolation(
                    fvarr=fvarr_bced, p=p[0], axis=1, pos="c"
                ),
                p=p[1],
                axis=2,
                pos="c",
            ),
            p=p[2],
            axis=3,
            pos="c",
        )
        assert out.shape == fvarr.shape
        return out

    def hydrofluxes(
        self, w: NamedNumpyArray, p: Tuple[int, int, int]
    ) -> Tuple[NamedNumpyArray, NamedNumpyArray, NamedNumpyArray]:
        """
        compute the Euler equation fluxes F, G, H with degree p polynomials
        args:
            w (NamedArray) : cell-averages of primitive variables. has shape (5, nx, ny, nz)
            p (Tuple[int, int, int]) : polynomial interpolation degree (px, py, pz)
        returns:
            F (NamedArray) : x-direction conservative fluxes. has shape (5, nx + 1, ny, nz)
            G (NamedArray) : y-direction conservative fluxes. has shape (5, nx, ny + 1, nz)
            H (NamedArray) : z-direction conservative fluxes. has shape (5, nx, ny, nz + 1)
        """
        pmax = max(p)
        gw = max(int(np.ceil(pmax / 2)) + 1, 2 * int(np.ceil(pmax / 2)))
        w_gw = self.bc.apply(w, gw=[gw, gw, gw])

        # interpolate x face midpoints
        w_xy = conservative_interpolation(w_gw, p=p[2], axis=3, pos="c")
        w_x = conservative_interpolation(w_xy, p=p[1], axis=2, pos="c")
        w_x_face_center_l = conservative_interpolation(w_x, p=p[0], axis=1, pos="l")
        w_x_face_center_r = conservative_interpolation(w_x, p=p[0], axis=1, pos="r")

        # interpolate y face midpoints
        w_yz = conservative_interpolation(w_gw, p=p[0], axis=1, pos="c")
        w_y = conservative_interpolation(w_yz, p=p[2], axis=3, pos="c")
        w_y_face_center_l = conservative_interpolation(w_y, p=p[1], axis=2, pos="l")
        w_y_face_center_r = conservative_interpolation(w_y, p=p[1], axis=2, pos="r")

        # interpolate z face midpoints
        w_zx = conservative_interpolation(w_gw, p=p[1], axis=2, pos="c")
        w_z = conservative_interpolation(w_zx, p=p[0], axis=1, pos="c")
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

        return F, G, H

    def hydrodynamics(self, u: NamedNumpyArray) -> Tuple[float, NamedNumpyArray]:
        """
        compute the Euler equation dynamics from cell-average conserved values u
        args:
            u (NamedNumpyArray) : cell-averages of conservative variables. has shape (5, nx, ny, nz)
        returns:
            dt, dudt (float, NamedNumpyArray) : time-step size, conservative variable dynamics
        """
        w = compute_primitives(u, gamma=self.gamma)
        if np.min(w.rho) < 0:
            raise BaseException("Negative density encountered.")
        if np.min(w.P) < 0:
            raise BaseException("Negative pressure encountered.")
        c_avg = compute_sound_speed(w=w, gamma=self.gamma)

        # timestep size dt
        if self.fixed_dt is None:
            dt = (
                self.CFL
                * 1
                / np.max(
                    (np.abs(w.vx) + c_avg) / self.h[0]
                    + (np.abs(w.vy) + c_avg) / self.h[1]
                    + (np.abs(w.vz) + c_avg) / self.h[2]
                ).item()
            )
        else:
            dt = self.fixed_dt

        if dt < 0:
            raise BaseException("Negative dt encountered.")

        # high-order fluxes
        F, G, H = self.hydrofluxes(w=w, p=self.p)
        dudt = -(1 / self.h[0]) * (F[:, 1:, ...] - F[:, :-1, ...])
        dudt += -(1 / self.h[1]) * (G[:, :, 1:, ...] - G[:, :, :-1, ...])
        dudt += -(1 / self.h[2]) * (H[:, :, :, 1:, ...] - H[:, :, :, :-1, ...])

        return dt, dudt

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
