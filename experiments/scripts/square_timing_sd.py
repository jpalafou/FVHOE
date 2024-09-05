import sys

sys.path.append("/home/jp7427/Desktop/spd/src")
sys.path.append("/home/jp7427/Desktop/spd/utils")

import initial_conditions as ic
from sdader_simulator import SDADER_Simulator

ndim = 2
N = 64
p = 3
n_steps = 100
cupy = True

sd = SDADER_Simulator(
    p=p,
    m=0,
    N=(N,) * ndim,
    cfl_coeff=0.01,
    init_fct=ic.step_function(),
    update="SD",
    use_cupy=cupy,
)
sd.perform_iterations(n_steps)
