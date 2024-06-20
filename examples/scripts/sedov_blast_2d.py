from functools import partial
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import shock_tube_2d
from fvhoe.solver import EulerSolver
import matplotlib.pyplot as plt
import numpy as np
import os

N = 128
p = 0


def snapshot_helper_function(s):
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
        im = s.plot_2d_slice(
            ax[idx], t=s.t, param=var, cmap="GnBu_r", z=0.5, verbose=False
        )
        fig.colorbar(im, ax=ax[idx], label=label)

    ax[0, 0].set_ylabel("$y$")
    ax[1, 0].set_ylabel("$y$")
    ax[1, 0].set_xlabel("$x$")
    ax[1, 1].set_xlabel("$x$")

    if not os.path.exists("snapshots/sedov-blast"):
        os.makedirs("snapshots/sedov-blast")

    plt.savefig(f"snapshots/sedov-blast/t={s.t:.2f}.png", dpi=300, bbox_inches="tight")


solver = EulerSolver(
    w0=partial(shock_tube_2d, radius=0.05, rho_in_out=(1, 1e-3), P_in_out=(1, 1e-5)),
    nx=N,
    ny=N,
    px=p,
    py=p,
    CFL=0.4,
    riemann_solver="hllc",
    bc=BoundaryCondition(x="free", y="free"),
    gamma=1.4,
    density_floor=False,
    pressure_floor=False,
    rho_P_sound_speed_floor=False,
    a_posteriori_slope_limiting=False,
    slope_limiter="minmod",
    cupy=True,
    snapshot_helper_function=snapshot_helper_function,
)

T = 0.2
solver.rkorder(
    T,
    downbeats=np.linspace(0, T, 21).tolist()[1:-1],
    filename="sedov-blast",
    overwrite=True,
)
