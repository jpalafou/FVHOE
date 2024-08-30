from itertools import product
from functools import partial
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
n_steps = 10
Ns = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
ps = [1, 3, 7]
cupys = [False, True]
save_path = "out/compare_sd_timing_square.csv"
first_order_integrator = True

data = []
for N, p, cupy in product(Ns, ps, cupys):
    print(f"-----fv, \n{N=}, {p=}, {cupy=}\n-----")
    # finite volume solver
    fv = EulerSolver(
        w0=partial(square, dims="xy", vx=1, vy=1),
        nx=N * (p + 1),
        ny=N * (p + 1),
        px=p,
        py=p,
        CFL=0.6,
        cupy=cupy,
    )
    if first_order_integrator:
        fv.euler(n=n_steps)
    else:
        fv.rkorder(n=n_steps)
    current_data = [
        dict(
            scheme="fv",
            N=N,
            p=p,
            nDOFs=N * (p + 1),
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
        N=(N, N),
        init_fct=ic.step_function(),
        cfl_coeff=0.6,
        update="SD",
        use_cupy=cupy,
    )
    sd.perform_iterations(n_steps)
    current_data += [
        dict(
            scheme="sd",
            N=N,
            p=p,
            nDOFs=N * (p + 1),
            steps=n_steps,
            substeps=1 if first_order_integrator else p + 1,
            cupy=cupy,
            execution_time=sd.execution_time,
        )
    ]
    data += current_data
    df = pd.DataFrame(data)
    df.to_csv(save_path, mode="w", index=False)
