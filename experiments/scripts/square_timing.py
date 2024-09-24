import cProfile
import pstats
import io
from fvhoe.initial_conditions import square
from fvhoe.solver import EulerSolver
from pstats import SortKey

ndim = 2
N = 2048
p = 7
n_steps = 100
cupy = True

fv = EulerSolver(
    w0=square(dims={2: "xy", 3: "xyz"}[ndim], vx=1, vy=1, vz={2: 0, 3: 1}[ndim]),
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

pr = cProfile.Profile()
pr.enable()

fv.euler(n=n_steps)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
