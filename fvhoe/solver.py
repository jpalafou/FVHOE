import numpy as np
from fvhoe.fv import fv_average, conservative_interpolation, transverse_reconstruction
from fvhoe.hydro import (
    compute_conservatives,
    compute_fluxes,
    compute_primitives,
    compute_sound_speed,
)
from fvhoe.initial_conditions import square
from fvhoe.ode import ODE
from fvhoe.riemann_solvers import advection_upwind, HLLC
from typing import Tuple, Union


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
        gamma: float = 5 / 3,
        bc: str = "periodic",
        riemann_solver: str = "HLLC",
        conservative_ic: bool = False,
    ):
        """
        ...
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
        self.n = nx, ny, nz
        self.h = hx, hy, hz
        self.p = px, py, pz
        self.CFL = CFL
        self.bc = bc

        # physics
        self.gamma = gamma

        # riemann solver
        if riemann_solver == "HLLC":
            self.riemann_solver = HLLC
        elif riemann_solver == "advection_upwind":
            self.riemann_solver = advection_upwind

        # integrator
        if conservative_ic:
            u0 = w0
        else:

            def u0(x, y, z):
                return compute_conservatives(w0(x, y, z), gamma=self.gamma)

        u0_fv = fv_average(f=u0, x=self.X, y=self.Y, z=self.Z, h=self.h, p=self.p)

        super().__init__(u0_fv)

    def f(self, t, u):
        return self.hydrodynamics(u)

    def snapshot(self):
        w = compute_primitives(self.u, gamma=self.gamma)
        log = {
            "rho": self.u[0].copy(),
            "E": self.u[1].copy(),
            "px": self.u[2].copy(),
            "py": self.u[3].copy(),
            "pz": self.u[4].copy(),
            "P": w[1].copy(),
            "vx": w[2].copy(),
            "vy": w[3].copy(),
            "vz": w[4].copy(),
        }
        self.snapshots[self.t] = log

    def apply_bcs(self, u: np.ndarray, bc_type: str, **kwargs) -> np.ndarray:
        """
        args:
            u (array_like) : array of cell parameters
            bc_type (str) : "periodic"
        returns:
            out (array_like) : u with additional padding for applied boundaries
        """
        if bc_type == "periodic":
            out = np.pad(u, mode="wrap", **kwargs)
        return out

    def hydrofluxes(
        self, w: np.ndarray, p: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        compute the Euler equation fluxes F, G, H with degree p polynomials
        args:
            w (array_like) : cell-averages of primitive variables. has shape (5, nx, ny, nz)
            p (Tuple[int, int, int]) : polynomial interpolation degree (px, py, pz)
        returns:
            F (array_like) : x-direction fluxes. has shape (5, nx + 1, ny, nz)
            G (array_like) : x-direction fluxes. has shape (5, nx, ny + 1, nz)
            H (array_like) : x-direction fluxes. has shape (5, nx, ny, nz + 1)
        """
        pmax = max(p)
        gw = max(int(np.ceil(pmax / 2)) + 1, 2 * int(np.ceil(pmax / 2)))
        w_gw = self.apply_bcs(
            w, bc_type=self.bc, pad_width=((0, 0), (gw, gw), (gw, gw), (gw, gw))
        )

        # interpolate x face midpoints
        w_xy = conservative_interpolation(w_gw, p=p[0], axis=3, pos="c")
        w_x = conservative_interpolation(w_xy, p=p[0], axis=2, pos="c")
        w_x_face_center_l = conservative_interpolation(w_x, p=p[0], axis=1, pos="l")
        w_x_face_center_r = conservative_interpolation(w_x, p=p[0], axis=1, pos="r")

        # interpolate y face midpoints
        w_yz = conservative_interpolation(w_gw, p=p[1], axis=1, pos="c")
        w_y = conservative_interpolation(w_yz, p=p[1], axis=3, pos="c")
        w_y_face_center_l = conservative_interpolation(w_y, p=p[1], axis=2, pos="l")
        w_y_face_center_r = conservative_interpolation(w_y, p=p[1], axis=2, pos="r")

        # interpolate z face midpoints
        w_zx = conservative_interpolation(w_gw, p=p[2], axis=2, pos="c")
        w_z = conservative_interpolation(w_zx, p=p[2], axis=1, pos="c")
        w_z_face_center_l = conservative_interpolation(w_z, p=p[2], axis=3, pos="l")
        w_z_face_center_r = conservative_interpolation(w_z, p=p[2], axis=3, pos="r")

        # pointwise numerical fluxes
        f_face_center = self.riemann_solver(
            wl=w_x_face_center_r[:, :-1, ...],
            wr=w_x_face_center_l[:, 1:, ...],
            gamma=self.gamma,
            dir="x",
        )
        g_face_center = self.riemann_solver(
            wl=w_y_face_center_r[:, :, :-1, ...],
            wr=w_y_face_center_l[:, :, 1:, ...],
            gamma=self.gamma,
            dir="y",
        )
        h_face_center = self.riemann_solver(
            wl=w_z_face_center_r[:, :, :, :-1, ...],
            wr=w_z_face_center_l[:, :, :, 1:, ...],
            gamma=self.gamma,
            dir="z",
        )

        # excess ghost zone counts after flux integral
        x_excess = (
            0,
            gw - int(np.ceil(p[0] / 2)) - 1,
            gw - 2 * int(np.ceil(p[0] / 2)),
            gw - 2 * int(np.ceil(p[0] / 2)),
        )
        y_excess = (
            0,
            gw - 2 * int(np.ceil(p[1] / 2)),
            gw - int(np.ceil(p[1] / 2)) - 1,
            gw - 2 * int(np.ceil(p[1] / 2)),
        )
        z_excess = (
            0,
            gw - 2 * int(np.ceil(p[2] / 2)),
            gw - 2 * int(np.ceil(p[2] / 2)),
            gw - int(np.ceil(p[2] / 2)) - 1,
        )

        # flux integrals
        F = transverse_reconstruction(
            transverse_reconstruction(f_face_center, axis=2, p=p[0]), axis=3, p=p[0]
        )[tuple(slice(x_excess[i] or None, -x_excess[i] or None) for i in range(4))]
        G = transverse_reconstruction(
            transverse_reconstruction(g_face_center, axis=1, p=p[1]), axis=3, p=p[1]
        )[tuple(slice(y_excess[i] or None, -y_excess[i] or None) for i in range(4))]
        H = transverse_reconstruction(
            transverse_reconstruction(h_face_center, axis=1, p=p[2]), axis=2, p=p[2]
        )[tuple(slice(z_excess[i] or None, -z_excess[i] or None) for i in range(4))]

        return F, G, H

    def hydrodynamics(self, u: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        compute the Euler equation dynamics from cell-average conserved values u
        args:
            u (array_like) : cell-averages of conservative variables. has shape (5, nx, ny, nz)
        returns:
            dt, dudt (tuple[float, array_like]) : time-step size, conservative variable dynamics
        """
        w = compute_primitives(u, gamma=self.gamma)
        if np.min(w[0]) < 0:
            raise BaseException("Negative density encountered.")
        if np.min(w[1]) < 0:
            raise BaseException("Negative pressure encountered.")
        c_avg = compute_sound_speed(w=w, gamma=self.gamma)

        # timestep size dt
        dts = []
        dtx = self.CFL * self.h[0] / np.max(np.abs(w[2, ...]) + c_avg)
        dty = self.CFL * self.h[1] / np.max(np.abs(w[3, ...]) + c_avg)
        dtz = self.CFL * self.h[2] / np.max(np.abs(w[4, ...]) + c_avg)
        dt = np.min([dtx, dty, dtz])
        if dt < 0:
            raise BaseException("Negative dt encountered.")

        # high-order fluxes
        F, G, H = self.hydrofluxes(w=w, p=self.p)
        dudt = -(1 / self.h[0]) * (F[:, 1:, ...] - F[:, :-1, ...])
        dudt += -(1 / self.h[1]) * (G[:, :, 1:, ...] - G[:, :, :-1, ...])
        dudt += -(1 / self.h[2]) * (H[:, :, :, 1:, ...] - H[:, :, :, :-1, ...])

        return dt, dudt

    def plot_1d_slice(
        self,
        ax,
        param: str,
        t: float = None,
        x=None,
        y=None,
        z=None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        if sum([x == None, y == None, z == None]) != 1:
            raise BaseException("One out of the three coordinates x-y-z must be None")
        t = max(self.snapshots.keys()) if t is None else t
        n = np.argmin(np.abs(np.array(list(self.snapshots.keys())) - t))
        t = list(self.snapshots.keys())[n]
        if x == None:
            j, k = np.argmin(np.abs(self.y - y)), np.argmin(np.abs(self.z - z))
            y, z = self.y[j], self.z[k]
            x = self.x
            x_for_plotting = self.x
            y_for_plotting = self.snapshots[t][param][:, j, k]
        elif y == None:
            i, k = np.argmin(np.abs(self.x - x)), np.argmin(np.abs(self.z - z))
            x, z = self.x[i], self.z[k]
            y = self.y
            x_for_plotting = self.y
            y_for_plotting = self.snapshots[t][param][i, :, k]
        elif z == None:
            i, j = np.argmin(np.abs(self.x - x)), np.argmin(np.abs(self.y - y))
            x, y = self.x[i], self.y[j]
            z = self.z
            x_for_plotting = self.z
            y_for_plotting = self.snapshots[t][param][i, j, :]
        if not verbose:
            message = ", ".join(
                [
                    f"{m:.2f}"
                    if (isinstance(m, int) or isinstance(m, float))
                    else f"[{m[0]:.2f},{m[-1]:.2f}]"
                    for m in [t, x, y, z]
                ]
            )
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

        ax.plot(x_for_plotting, y_for_plotting, **kwargs)

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
