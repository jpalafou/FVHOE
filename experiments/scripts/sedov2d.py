from functools import partial
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import sedov
from fvhoe.solver import EulerSolver
import matplotlib.pyplot as plt
import os

# argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-N", "--N", type=int, required=True)
parser.add_argument("-p", "--p", type=int, required=True)
parser.add_argument("--NAD", type=float, default=0.01)
parser.add_argument("--SED", action="store_true")
parser.add_argument("-C", "--CFL", type=float, default=0.8)
parser.add_argument("--cupy", action="store_true")
parser.add_argument("-r", "--riemann", type=str, default="hllc")
parser.add_argument("-T", type=float, default=1.0)
args = parser.parse_args()

# solver params
CFL = args.CFL
cupy = args.cupy
N = args.N
NAD = args.NAD
p = args.p
riemann = args.riemann
SED = args.SED
T = args.T

# data management
snapshot_parent_dir = "/scratch/gpfs/jp7427/fvhoe/snapshots/"
project_name = f"sedov2d_{CFL=}_{cupy=}_{N=}_{NAD=}_{p=}_{riemann=}_{SED=}_{T=}"


def snapshot_helper_function(s):
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    ax[0, 0].set_title(r"$\rho$")
    s.plot_2d_slice(ax[0, 0], param="rho", t=s.t, z=0.5)
    ax[0, 1].set_title(r"$P$")
    s.plot_2d_slice(ax[0, 1], param="P", t=s.t, z=0.5)
    ax[0, 2].set_title(r"$v$")
    s.plot_2d_slice(ax[0, 2], param="v", t=s.t, z=0.5)
    ax[1, 0].set_title(r"$v_x$")
    s.plot_2d_slice(ax[1, 0], param="vx", t=s.t, z=0.5)
    ax[1, 1].set_title(r"$v_y$")
    s.plot_2d_slice(ax[1, 1], param="vy", t=s.t, z=0.5)
    ax[1, 2].set_title(r"$v_z$")
    s.plot_2d_slice(ax[1, 2], param="vz", t=s.t, z=0.5)

    image_dir = "out/" + project_name
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    plt.savefig(image_dir + f"/t={s.t:.2f}.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(s.timeseries_E)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$E$")
    plt.savefig(image_dir + f"/E_t={s.t:.2f}.png", dpi=300, bbox_inches="tight")


# set up solver
solver = EulerSolver(
    w0=partial(sedov, dims="xy"),
    fv_ic=True,
    conservative_ic=True,
    bc=BoundaryCondition(x=("reflective", "outflow"), y=("reflective", "outflow")),
    CFL=CFL,
    x=(0, 1.1),
    y=(0, 1.1),
    nx=N,
    ny=N,
    px=p,
    py=p,
    riemann_solver=riemann,
    gamma=1.4,
    all_floors=True,
    a_posteriori_slope_limiting=p > 0,
    NAD=NAD,
    SED=SED,
    slope_limiter="minmod",
    cupy=cupy,
    snapshot_helper_function=snapshot_helper_function,
)

# run simulation
solver.rkorder(T=T, snapshot_dir=snapshot_parent_dir + project_name)
