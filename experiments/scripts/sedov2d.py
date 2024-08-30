from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import sedov
from fvhoe.scripting import EulerSolver_wrapper
from functools import partial

N = 256
p = 3

# set up solver
solver_config = dict(
    fv_ic=True,
    conservative_ic=True,
    x=(0, 1.1),
    y=(0, 1.1),
    nx=N,
    ny=N,
    px=p,
    py=p,
    gamma=1.4,
    a_posteriori_slope_limiting=p > 0,
    all_floors=True,
    NAD=1e-2,
    cupy=False,
)

# run solver
EulerSolver_wrapper(
    project_pref="sedov-2D",
    snapshot_parent_dir="/scratch/gpfs/jp7427/fvhoe/snapshots",
    summary_parent_dir="out",
    ic=partial(sedov, dims="xy"),
    bc=BoundaryCondition(x=("reflective", "outflow"), y=("reflective", "outflow")),
    T=1.0,
    **solver_config,
)