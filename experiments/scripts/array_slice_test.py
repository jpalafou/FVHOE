from functools import partial
import numpy as np

# array manager import
from fvhoe.array_manager import ArrayManager, get_array_slice
from fvhoe.config import conservative_names

# named array imporrt
from fvhoe.named_array import NamedNumpyArray, NamedCupyArray
from fvhoe.fv import get_view

N = 32
mode = "both"
cupy = False

# simplify names
gv = partial(get_view, ndim=3)
gas = get_array_slice

arr = 10 * np.random.rand(5, N, N, N)

if mode in ("am", "both"):
    am = ArrayManager()
    if cupy:
        am.enable_cupy()
    am.add("u", arr)

if mode in ("nnp", "both"):
    if cupy:
        arr_nnp = NamedCupyArray(arr, conservative_names)
    else:
        arr_nnp = NamedNumpyArray(arr, conservative_names)

# main experiment
for _ in range(1000):
    if mode in ("am", "both"):
        arr_am = am.get("u")
        arr_am[gas("rho", x=(2, -2))] -= 1e-12 * np.square(
            arr_am[gas("rho", x=(2, -2))]
        )

    if mode in ("nnp", "both"):
        arr_nnp.rho[gv(axis=0, cut=(2, 2))] -= 1e-12 * np.square(
            arr_nnp.rho[gv(axis=0, cut=(2, 2))]
        )

# check results
if mode == "both":
    print(np.mean(np.abs(arr_nnp.rho - arr_am[gas("rho")])))
