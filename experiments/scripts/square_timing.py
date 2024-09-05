from fvhoe.initial_conditions import square
from fvhoe.solver import EulerSolver

ndim = 2
N = 64
p = 3
n_steps = 100
cupy = True

fv = EulerSolver(
    w0=square(dims="xy", vx=1, vy=1, vz={2: 0, 3: 1}[ndim]),
    nx=N,
    ny=N,
    nz={2: 1, 3: N}[ndim],
    px=p,
    py=p,
    pz={2: 0, 3: p}[ndim],
    riemann_solver="llf",
    CFL=0.01,
    cupy=cupy,
)
fv.euler(n=n_steps)
