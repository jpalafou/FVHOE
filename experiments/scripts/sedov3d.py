from functools import partial
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import sedov
from fvhoe.solver import EulerSolver
from itertools import product
import matplotlib.pyplot as plt

# sedov blast params
gamma = 1.4

for t, N, p, rs, lc in product([0.7], [32], [1], ["hllc"], [dict(NAD=1e-5)]):
    # savepath
    snapshot_dir = "/scratch/gpfs/jp7427/fvhoe/snapshots/"
    snapshot_dir += f"sedov3d_{t=}_{N=}_{p=}_{rs=}_{lc=}"

    # run solver
    solver = EulerSolver(
        w0=partial(sedov, dims="xyz", mode="corner"),
        conservative_ic=True,
        fv_ic=True,
        gamma=gamma,
        bc=BoundaryCondition(
            x=("reflective", "outflow"),
            y=("reflective", "outflow"),
            z=("reflective", "outflow"),
        ),
        CFL=0.8,
        nx=N,
        ny=N,
        nz=N,
        px=p,
        py=p,
        pz=p,
        riemann_solver=rs,
        all_floors=True,
        snapshots_as_fv_averages=False,
        cupy=True,
        a_posteriori_slope_limiting=True,
        **lc,
    )
    solver.rkorder(t, snapshot_dir=snapshot_dir)

    # plot in snapshot path
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)

    indexing = dict(z=0.5, t=t)

    # density
    ax[0, 0].set_title(r"$\rho$")
    solver.plot_2d_slice(ax[0, 0], param="rho", **indexing)

    # pressure
    ax[0, 1].set_title(r"$P$")
    solver.plot_2d_slice(ax[0, 1], param="P", verbose=False, **indexing)

    # velocity magnitude
    ax[0, 2].set_title(r"$v$")
    solver.plot_2d_slice(ax[0, 2], param="v", verbose=False, **indexing)

    # velocity components
    ax[1, 0].set_title(r"$v_x$")
    solver.plot_2d_slice(ax[1, 0], param="vx", verbose=False, **indexing)

    ax[1, 1].set_title(r"$v_y$")
    solver.plot_2d_slice(ax[1, 1], param="vy", verbose=False, **indexing)

    ax[1, 2].set_title(r"$v_z$")
    solver.plot_2d_slice(ax[1, 2], param="vz", verbose=False, **indexing)

    fig.savefig(snapshot_dir + "/plot.png", dpi=300, bbox_inches="tight")
