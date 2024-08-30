from functools import lru_cache
import numpy as np
from typing import Tuple

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

conservative_map = {"rho": 0, "E": 1, "mx": 2, "my": 3, "mz": 4}
primitive_map = {"rho": 0, "P": 1, "vx": 2, "vy": 3, "vz": 4}


@lru_cache(maxsize=100)
def get_array_slice(
    var: str = None,
    x: Tuple[int, int] = None,
    y: Tuple[int, int] = None,
    z: Tuple[int, int] = None,
    var_idx: int = None,
    conservative: bool = False,
    ndim: int = 4,
) -> tuple:
    """
    Get the slice for the given variable and coordinates for an array with the following axes:
        axis 0: variable
        axis 1: x-coordinate
        axis 2: y-coordinate
        axis 3: z-coordinate
    args:
        var (str) : variable name
        x (Tuple[int, int]) : x-coordinate slice
        y (Tuple[int, int]) : y-coordinate slice
        z (Tuple[int, int]) : z-coordinate slice
        var_idx (int) : variable index. If var is None, use var_idx
        conservative (bool) : whether to use conservative variables
            True: use conservative variables (rho, E, mx, my, mz)
            False: use primitive variables (rho, P, vx, vy, vz)
        ndim (int) : number of dimensions of the array
    returns:
        tuple : slices for the given variable and coordinates
    """
    slices = [slice(None)] * ndim
    if var is not None:
        slices[0] = conservative_map[var] if conservative else primitive_map[var]
    elif var_idx is not None:
        slices[0] = var_idx
    if x is not None:
        slices[1] = slice(x[0] or None, x[1] or None)
    if y is not None:
        slices[2] = slice(y[0] or None, y[1] or None)
    if z is not None:
        slices[3] = slice(z[0] or None, z[1] or None)
    return tuple(slices)


class ArrayManager:
    def __init__(
        self,
    ):
        self.arrays = {}
        self.using_cupy = False

    def enable_cupy(self):
        if CUPY_AVAILABLE:
            self.using_cupy = True
        else:
            print("WARNING: CuPy is not available. Falling back to NumPy.")

    def disable_cupy(self):
        self.using_cupy = False

    def add(self, name: str, array: np.ndarray):
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be of type numpy")
        if self.using_cupy:
            array = cp.asarray(array)
        self.arrays[name] = array

    def _check_name(self, name: str):
        if name not in self.arrays:
            raise KeyError(f"Array with name '{name}' not found.")

    def convert_to_cupy(self, name: str):
        self._check_name(name)
        if not self.using_cupy:
            raise ValueError("CuPy is not enabled.")
        if isinstance(self.arrays[name], np.ndarray):
            self.arrays[name] = cp.asarray(self.arrays[name])

    def convert_to_numpy(self, name: str):
        self._check_name(name)
        if not self.using_cupy:
            raise ValueError("CuPy is not enabled.")
        if CUPY_AVAILABLE and isinstance(self.arrays[name], cp.ndarray):
            self.arrays[name] = cp.asnumpy(self.arrays[name])

    def remove(self, name: str):
        self._check_name(name)
        del self.arrays[name]

    def get(self, name: str, return_numpy: bool = False) -> np.ndarray:
        self._check_name(name)
        return self.arrays[name]

    def get_numpy(self, name: str) -> np.ndarray:
        self._check_name(name)
        if self.using_cupy:
            return cp.asnumpy(self.arrays[name])
        return self.arrays[name]
