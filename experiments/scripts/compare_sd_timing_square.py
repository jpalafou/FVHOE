from itertools import product
from fvhoe.initial_conditions import square
from fvhoe.solver import EulerSolver
import os
import numpy as np
import pandas as pd
import sys

# experiment params
n_dims = 3
n_steps = 1
if n_dims == 1:
    DOFss = 2 ** np.arange(3, 25)
elif n_dims == 2:
    DOFss = 2 ** np.arange(3, 13)
elif n_dims == 3:
    DOFss = 2 ** np.arange(3, 8)
ps = [1, 3, 7]
slope_limitings = [False, True]
NAD_tol = 1e-5
NAD_range = "relative"
save_path = f"out/square_timing_{n_dims}D.csv"
CFL = 0.01
cupys = [False, True]
numpy_DOFs_max = 128**2
include_sd = True
first_order_integrator = True

# add spd to path
if include_sd:
    spd_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "spd", "src")
    )

    if spd_path not in sys.path:
        sys.path.append(spd_path)

    import initial_conditions as ic
    from sdader_simulator import SDADER_Simulator

# run and time
os.makedirs("out", exist_ok=True)
data = []
for DOFs, p, slope_limiting, cupy in product(DOFss, ps, slope_limitings, cupys):
    # skip if too many DOFs for numpy
    if not cupy and DOFs**n_dims > numpy_DOFs_max:
        continue

    print(f"-----fv, \n{DOFs=}, {p=}, {slope_limiting=}, {cupy=}\n-----")
    # finite volume solver
    fv = EulerSolver(
        w0=square(
            dims={1: "x", 2: "xy", 3: "xyz"}[n_dims],
            vx=1,
            vy={1: 0, 2: 1, 3: 1}[n_dims],
            vz={1: 0, 2: 0, 3: 1}[n_dims],
        ),
        nx=DOFs,
        ny=DOFs if n_dims > 1 else 1,
        nz=DOFs if n_dims > 2 else 1,
        px=p,
        py=p if n_dims > 1 else 0,
        pz=p if n_dims > 2 else 0,
        riemann_solver="llf",
        CFL=CFL,
        a_posteriori_slope_limiting=slope_limiting,
        NAD=NAD_tol,
        NAD_mode="local",
        NAD_range=NAD_range,
        cupy=cupy,
    )
    if first_order_integrator:
        fv.euler(n=n_steps)
    else:
        fv.rkorder(n=n_steps)
    data.append(
        dict(
            scheme="fv",
            n_dims=n_dims,
            DOFs_per_dim=DOFs,
            DOFs=DOFs**n_dims,
            p=p,
            slope_limiting=slope_limiting,
            NAD_tol=NAD_tol,
            NAD_range=NAD_range,
            n_steps=n_steps,
            n_substeps=1 if first_order_integrator else min(p + 1, 4),
            cupy=cupy,
            total_time=fv.timer.cum_time["TOTAL"],
            riemann_time=fv.timer.cum_time["(high-order) riemann solver"]
            + fv.timer.cum_time["(fallback scheme) riemann solver"],
            interpolation_time=fv.timer.cum_time[
                "(high-order) conservative interpolation"
            ]
            + fv.timer.cum_time["(high-order) transverse reconstruction"]
            + fv.timer.cum_time["(fallback scheme) conservative interpolation"]
            + fv.timer.cum_time["(fallback scheme) transverse reconstruction"],
        )
    )

    # run sd solver if it's included
    if include_sd:
        print(f"-----sd, \n{DOFs=}, {p=}, {slope_limiting=}, {cupy=}\n-----")
        # finite element solver
        sd = SDADER_Simulator(
            p=p,
            m=0 if first_order_integrator else -1,
            N=(DOFs // (p + 1),) * n_dims,
            init_fct=ic.step_function(),
            cfl_coeff=CFL,
            update="FV" if slope_limiting else "SD",
            FB=slope_limiting,
            tolerance=NAD_tol,
            NAD={"relative": "delta", "absolute": ""}[NAD_range],
            use_cupy=cupy,
        )
        sd.perform_iterations(n_steps)
        data.append(
            dict(
                scheme="sd",
                n_dims=n_dims,
                DOFs_per_dim=DOFs,
                DOFs=DOFs**n_dims,
                p=p,
                slope_limiting=slope_limiting,
                NAD_tol=NAD_tol,
                NAD_range=NAD_range,
                n_steps=n_steps,
                n_substeps=1 if first_order_integrator else p + 1,
                cupy=cupy,
                total_time=sd.timer.cum_time["TOTAL"],
                riemann_time=sd.timer.cum_time["(sd) riemann solver"]
                + sd.timer.cum_time["(fv) riemann solver"],
                interpolation_time=sd.timer.cum_time["(sd) interpolate"],
            )
        )

    df = pd.DataFrame(data)
    df.to_csv(save_path, mode="w", index=False)
