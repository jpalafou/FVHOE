from functools import partial
from fvhoe.array_slice import array_slice_generator, conservative_map
from fvhoe.config import conservative_names
from fvhoe.fv import get_view
from fvhoe.named_array import NamedNumpyArray
import numpy as np

N = 128

# initialize arrays
arr = np.random.rand(5, N, N, N)
arr_np = arr.copy()
arr_nnp = NamedNumpyArray(arr_np, conservative_names)

# simplify names
gv = partial(get_view, ndim=3)
asg = array_slice_generator
cm = conservative_map

# main experiment
for _ in range(1000):
    arr_nnp.rho[gv(axis=0, cut=(2, 2))] -= 1e-6 * np.square(
        arr_nnp.rho[gv(axis=0, cut=(2, 2))]
    )
    arr_np[asg(cm["rho"], x=(2, -2))] -= 1e-6 * np.square(
        arr_np[asg(cm["rho"], x=(2, -2))]
    )

# check results
print(np.mean(np.abs(arr_nnp.rho - arr_np[asg(cm["rho"])])))
