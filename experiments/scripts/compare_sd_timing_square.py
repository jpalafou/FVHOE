from itertools import product
from fvhoe.initial_conditions import square
from fvhoe.solver import EulerSolver
import pandas as pd

# import sd code
import sys

sys.path.append("/home/jp7427/Desktop/spd/src")
sys.path.append("/home/jp7427/Desktop/spd/utils")

import initial_conditions as ic
from sdader_simulator import SDADER_Simulator

# experiment params
n_dims = 2
n_steps = 100
Ns = [2048]
ps = [7]
cupys = [True]
first_order_integrator = True
save_path = f"out/compare_sd_timing_square_{n_dims=}.csv"

data = []
for N, p, cupy in product(Ns, ps, cupys):
    if not cupy and N > 256:
        continue
    print(f"-----fv, \n{N=}, {p=}, {cupy=}\n-----")
    # finite volume solver
    fv = EulerSolver(
        w0=square(
            dims={2: "xy", 3: "xyz"}[n_dims], vx=1, vy=1, vz={2: 0, 3: 1}[n_dims]
        ),
        nx=N,
        ny=N,
        nz={2: 1, 3: N}[n_dims],
        px=p,
        py=p,
        pz={2: 0, 3: p}[n_dims],
        riemann_solver="llf",
        CFL=0.01,
        cupy=cupy,
    )
    if first_order_integrator:
        fv.euler(n=n_steps)
    else:
        fv.rkorder(n=n_steps)
    current_data = [
        dict(
            n_dims=n_dims,
            scheme="fv",
            N=N,
            p=p,
            nDOFs=N**n_dims,
            steps=n_steps,
            substeps=1 if first_order_integrator else min(p + 1, 4),
            cupy=cupy,
            execution_time=fv.execution_time,
        )
    ]

    print(f"-----sd, \n{N=}, {p=}, {cupy=}\n-----")
    # finite element solver
    sd = SDADER_Simulator(
        p=p,
        m=0 if first_order_integrator else -1,
        N=(N // (p + 1),) * n_dims,
        init_fct=ic.step_function(),
        cfl_coeff=0.01,
        update="SD",
        use_cupy=cupy,
    )
    sd.perform_iterations(n_steps)
    current_data += [
        dict(
            n_dims=n_dims,
            scheme="sd",
            N=N // (p + 1),
            p=p,
            nDOFs=N**n_dims,
            steps=n_steps,
            substeps=1 if first_order_integrator else p + 1,
            cupy=cupy,
            execution_time=sd.execution_time,
        )
    ]
    data += current_data
    df = pd.DataFrame(data)
    df.to_csv(save_path, mode="w", index=False)
