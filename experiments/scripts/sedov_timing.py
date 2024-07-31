from itertools import product
from functools import partial
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import sedov
from fvhoe.solver import EulerSolver
import pandas as pd

# experiment params
dims = "xyz"
n = 10
Ns = [16, 32, 64, 128, 256][-1:]
ps = [0, 1, 2, 3, 4, 5, 6, 7, 8][-1:]
cupys = [True]

data = []
for N, p, cupy in product(Ns, ps, cupys):
    print(f"-----\n{N=}, {p=}, {cupy=}\n-----")
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
    data.append(
        dict(
            dims=dims,
            N=N,
            p=p,
            integrator=solver.integrator,
            n_iterations=n,
            cupy=cupy,
            execution_time=solver.execution_time,
        )
    )

    pd.DataFrame(data).to_csv(f"data/sedov_{dims}_timing.csv", index=False)
