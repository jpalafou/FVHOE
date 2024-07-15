from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import kelvin_helmholtz_2d
from fvhoe.solver import EulerSolver
import matplotlib.pyplot as plt
import numpy as np
import os

T = 0.8
n_snapshots = 2
N = 2048
p = 3
NAD = 1e-2
SED = False
snapshot_dir = f"kelvin-helmholtz_{N=}_{p=}_{NAD=}_{SED=}"
snapshot_dir += "/scratch/gpfs/jp7427/fvhoe/snapshots/"


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

    if not os.path.exists(f"snapshots/{snapshot_dir}"):
        os.makedirs(f"snapshots/{snapshot_dir}")

    plt.savefig(
        f"snapshots/{snapshot_dir}/t={s.t:.2f}.png", dpi=300, bbox_inches="tight"
    )


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
    slope_limiter="minmod",
    NAD=NAD,
    SED=SED,
    cupy=True,
    snapshot_helper_function=snapshot_helper_function,
)

solver.rkorder(
    T=T,
    downbeats=np.linspace(0, T, n_snapshots),
    snapshot_dir=snapshot_dir,
)
