import cupy as cp
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.config import conservative_names
from fvhoe.initial_conditions import double_mach_reflection_2d, variable_array
from fvhoe.scripting import EulerSolver_wrapper
from itertools import product
import numpy as np
import os

# job array index
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

# configure NAD
NAD_values = [1.0, 0.1]  # [1e-2, 1e-3, 1e-5]
NAD_mode_values = ["local"]
NAD_range_values = ["relative"]  # ["relative", "absolute"]
NAD_vars_values = [["rho", "P", "vx", "vy"]]
convex = [False, True]
NAD_configs = [
    {
        "NAD": NAD,
        "NAD_mode": NAD_mode,
        "NAD_range": NAD_range,
        "NAD_vars": NAD_vars,
        "convex": convex,
    }
    for NAD, NAD_mode, NAD_range, NAD_vars, convex in product(
        NAD_values, NAD_mode_values, NAD_range_values, NAD_vars_values, convex
    )
]

print(NAD_configs[idx])

# other parameters
Nx = 960
p = 8

# set up solver
solver_config = dict(
    x=(0, 4),
    nx=Nx,
    ny=Nx // 4,
    px=p,
    py=p,
    CFL=0.6,
    gamma=1.4,
    a_posteriori_slope_limiting=p > 0,
    density_floor=1e-16,
    pressure_floor=1e-16,
    cupy=True,
    **NAD_configs[idx],
)


def upper_bc(x, y, z, t):
    """
    dirichlet boundary at y=0 for double mach reflection
    """
    theta = np.pi / 3
    xp = (10 * t / np.sin(theta)) + (1 / 6) + (y / np.tan(theta))
    out = cp.asarray([np.empty_like(x)] * 5), conservative_names

    # primitive
    rho = np.where(x < xp, 8.0, 1.4)
    vx = np.where(x < xp, 7.145, 0.0)
    vy = np.where(x < xp, -8.25 / 2, 0.0)
    P = np.where(x < xp, 116.5, 1.0)

    # conservative
    mx, my = rho * vx, rho * vy
    out = variable_array(
        shape=x.shape,
        rho=rho,
        P=P / (1.4 - 1) + 0.5 * (mx * vx + my * vy),
        vx=mx,
        vy=my,
        vz=0.0,
        conservative=True,
    )

    return out


# run solver
EulerSolver_wrapper(
    project_pref="dmr",  # "double-mach-reflection"
    snapshot_parent_dir="/scratch/gpfs/jp7427/fvhoe/snapshots",
    summary_parent_dir="out",
    ic=double_mach_reflection_2d,
    bc=BoundaryCondition(
        x=("dirichlet", "outflow"),
        x_value=(
            np.array(
                [
                    8.0,
                    116.5 / (1.4 - 1) + 0.5 * 8.0 * (7.145**2 + (-8.25 / 2) ** 2),
                    8.0 * 7.145,
                    8.0 * (-8.25 / 2),
                    0,
                ]
            ),
            None,
        ),
        y=(
            "special-case-double-mach-reflection-y=0",
            "dirichlet",
        ),
        y_value=(None, upper_bc),
    ),
    T=0.2,
    integrator=3,
    plot_kwargs=dict(
        z=0.5,
        contour=True,
        levels=np.linspace(1.5, 22.9705, 30),
        colors="k",
        linewidths=0.25,
    ),
    **solver_config,
)
