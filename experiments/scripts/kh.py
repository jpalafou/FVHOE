from fvhoe.initial_conditions import kelvin_helmholtz_2d
from fvhoe.scripting import EulerSolver_wrapper

N = 256
p = 3

# set up solver
solver_config = dict(
    nx=N,
    ny=N,
    px=p,
    py=p,
    gamma=1.4,
    a_posteriori_slope_limiting=p > 0,
    NAD=1e-2,
    cupy=True,
)

# run solver
EulerSolver_wrapper(
    project_pref="kelvin-helmholtz",
    snapshot_parent_dir="/scratch/gpfs/jp7427/fvhoe/snapshots",
    summary_parent_dir="out",
    ic=kelvin_helmholtz_2d,
    n_snapshots=5,
    T=0.8,
    **solver_config,
)
