from itertools import product
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import kelvin_helmholtz_2d
from fvhoe.solver import EulerSolver
import pandas as pd

n_timesteps = 10

data = []
for N, p, cupy in product([32, 64, 128, 256, 512], [0, 1, 2, 3], [False, True]):
    solver = EulerSolver(
        w0=kelvin_helmholtz_2d,
        nx=N,
        ny=N,
        px=p,
        py=p,
        CFL=0.8,
        riemann_solver="hllc",
        bc=BoundaryCondition(x="periodic", y="periodic"),
        gamma=1.4,
        a_posteriori_slope_limiting=True,
        density_floor=False,
        pressure_floor=False,
        rho_P_sound_speed_floor=False,
        slope_limiter="minmod",
        progress_bar=False,
        cupy=cupy,
    )
    solver.rkorder(n_timesteps=n_timesteps)

    # save data
    data.append(
        dict(
            N=N,
            p=p,
            steps=solver.step_count,
            f_count=solver.f_evaluation_count,
            cupy=cupy,
            time=solver.execution_time,
        )
    )
    df = pd.DataFrame(data)
    df.to_csv("data/kh_Np_timing.csv", index=False)
