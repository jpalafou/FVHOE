from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import kelvin_helmholtz_2d
from fvhoe.solver import EulerSolver
import matplotlib.pyplot as plt
import numpy as np

N = 256
p = 3
T = 0.1
filename = f"kh_timing_{N=}_{p=}"

solver = EulerSolver(
    w0=kelvin_helmholtz_2d,
    nx=N,
    ny=N,
    px=p,
    py=p,
    CFL=0.8,
    riemann_solver="hllc",
    bc=BoundaryCondition(x="periodic", y="periodic"),
    gamma=1.4,
    a_posteriori_slope_limiting=True,
    density_floor=False,
    pressure_floor=False,
    rho_P_sound_speed_floor=False,
    slope_limiter="minmod",
    cupy=True,
)

solver.rkorder(
    T,
    downbeats=np.linspace(0, T, 21),
    filename=filename,
    overwrite=True,
)

# make plot
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
for var, label, idx in zip(
    ["rho", "P", "vx", "vy"],
    [
        r"$\overline{\rho}$",
        r"$\overline{P}$",
        r"$\overline{v}_x$",
        r"$\overline{v}_y$",
    ],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
):
    im = solver.plot_2d_slice(
        ax[idx], t=T, param=var, cmap="GnBu_r", z=0.5, verbose=False
    )
    fig.colorbar(im, ax=ax[idx], label=label)

ax[0, 0].set_ylabel("$y$")
ax[1, 0].set_ylabel("$y$")
ax[1, 0].set_xlabel("$x$")
ax[1, 1].set_xlabel("$x$")

plt.savefig(f"snapshots/{filename}/plot.png", dpi=300, bbox_inches="tight")
