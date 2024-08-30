from functools import partial
from fvhoe.initial_conditions import square
from fvhoe.solver import EulerSolver
import matplotlib.pyplot as plt

N = 512
p = 3

fv = EulerSolver(
    w0=partial(square, dims="xy", vx=1, vy=1),
    nx=N,
    ny=N,
    px=p,
    py=p,
    CFL=0.8,
    cupy=True,
)
fv.rkorder(
    1.0,
    snapshot_dir=f"/scratch/gpfs/jp7427/fvhoe/snapshots/square_timing_test_{N=}_{p=}",
)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
fv.plot_2d_slice(axs[0, 0], param="rho")
fv.plot_2d_slice(axs[0, 1], param="P")
fv.plot_2d_slice(axs[1, 0], param="vx")
fv.plot_2d_slice(axs[1, 1], param="vy")
axs[0, 0].set_ylabel(r"$y$")
axs[1, 0].set_ylabel(r"$y$")
axs[1, 0].set_ylabel(r"$x$")
axs[1, 1].set_ylabel(r"$x$")
fig.savefig("/home/jp7427/Desktop/FVHOE/out/square.png")
