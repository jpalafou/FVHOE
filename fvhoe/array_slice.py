from functools import lru_cache
from typing import Tuple

conservative_map = {"rho": 0, "E": 1, "mx": 2, "my": 3, "mz": 4}


@lru_cache(maxsize=100)
def array_slice_generator(
    var: int = None,
    x: Tuple[int, int] = None,
    y: Tuple[int, int] = None,
    z: Tuple[int, int] = None,
) -> int:
    slices = [slice(None), slice(None), slice(None), slice(None)]
    if var is not None:
        slices[0] = var
    if x is not None:
        slices[1] = slice(x[0] or None, x[1] or None)
    if y is not None:
        slices[2] = slice(y[0] or None, y[1] or None)
    if z is not None:
        slices[3] = slice(z[0] or None, z[1] or None)
    return tuple(slices)
