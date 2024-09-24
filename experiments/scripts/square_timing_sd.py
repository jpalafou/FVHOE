import cProfile
import pstats
import io
import os
from pstats import SortKey
import sys

ndim = 2
N = 2048
p = 7
n_steps = 100
cupy = True

# add spd to path
spd_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "spd", "src")
)

if spd_path not in sys.path:
    sys.path.append(spd_path)

import initial_conditions as ic  # noqa: E402
from sdader_simulator import SDADER_Simulator  # noqa: E402

sd = SDADER_Simulator(
    p=p,
    m=0,
    N=(N // (p + 1),) * ndim,
    cfl_coeff=0.01,
    init_fct=ic.step_function(),
    update="SD",
    use_cupy=cupy,
)

pr = cProfile.Profile()
pr.enable()

sd.perform_iterations(n_steps)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
