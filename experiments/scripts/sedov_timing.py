from itertools import product
from functools import partial
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import sedov
from fvhoe.solver import EulerSolver
import pandas as pd

# experiment params
dims = "xy"
n = 10
Ns = [16, 32, 64, 128, 256, 512, 1024]
ps = [0, 1, 2, 3, 4, 5, 6, 7, 8]
cupys = [False, True]
save_path = f"out/sedov_{dims}_timing.csv"
append = True

data = []
for N, p, cupy in product(Ns, ps, cupys):
    print(f"-----\n{N=}, {p=}, {cupy=}\n-----")
    if append:
        df = pd.read_csv(save_path)
    solver = EulerSolver(
        w0=partial(sedov, dims=dims),
        bc=BoundaryCondition(
            x=("reflective", "outflow") if "x" in dims else None,
            y=("reflective", "outflow") if "y" in dims else None,
            z=("reflective", "outflow") if "z" in dims else None,
        ),
        gamma=1.4,
        conservative_ic=True,
        fv_ic=True,
        nx=N if "x" in dims else 1,
        ny=N if "y" in dims else 1,
        nz=N if "z" in dims else 1,
        px=p if "x" in dims else 0,
        py=p if "y" in dims else 0,
        pz=p if "z" in dims else 0,
        riemann_solver="hllc",
        a_posteriori_slope_limiting=p > 0,
        NAD=1e-3,
        all_floors=True,
        snapshots_as_fv_averages=False,
        cupy=cupy,
    )

    # run simulation
    solver.rkorder(n=n)

    # add data
    current_data = dict(
        dims=dims,
        N=N,
        p=p,
        integrator=solver.integrator,
        n_iterations=n,
        cupy=cupy,
        execution_time=solver.execution_time,
    )

    if append:
        df = pd.concat([df, pd.DataFrame([current_data])])
    else:
        data.append(current_data)
        df = pd.DataFrame(data)
    df.to_csv(save_path, mode="w", index=False)
