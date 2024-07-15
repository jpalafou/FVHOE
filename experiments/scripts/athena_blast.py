from fvhoe.initial_conditions import athena_blast
from fvhoe.solver import EulerSolver
import matplotlib.pyplot as plt

# save path
snapshot_dir = "/scratch/gpfs/jp7427/fvhoe/snapshots/athena_blast/"

# run solver
solver = EulerSolver(
    w0=athena_blast,
    gamma=5 / 3,
    CFL=0.8,
    x=(-0.5, 0.5),
    y=(-0.75, 0.75),
    nx=400,
    ny=600,
    px=2,
    py=2,
    riemann_solver="hllc",
    a_posteriori_slope_limiting=True,
    NAD=1e-2,
    all_floors=True,
    snapshots_as_fv_averages=False,
    cupy=True,
)
solver.rkorder(1.5, downbeats=[0.2], snapshot_dir=snapshot_dir)

# plot in snapshot path
fig, ax = plt.subplots()
solver.plot_2d_slice(ax, param="rho", z=0.5)
fig.savefig(snapshot_dir + "/plot.png", dpi=300, bbox_inches="tight")
