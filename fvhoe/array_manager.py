from functools import lru_cache
import numpy as np
from typing import Tuple, Union

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

VARIABLE_IDX_MAP = {
    "rho": 0,
    "P": 1,
    "vx": 2,
    "vy": 3,
    "vz": 4,
    "E": 1,
    "mx": 2,
    "my": 3,
    "mz": 4,
}


@lru_cache(maxsize=100)
def get_array_slice(
    var: Union[str, Tuple[str]] = None,
    x: Tuple[int, int] = None,
    y: Tuple[int, int] = None,
    z: Tuple[int, int] = None,
    ndim: int = 4,
    axis: int = None,
    cut: Tuple[int, int] = None,
    step: int = 1,
) -> tuple:
    """
    Get the slice for the given variable and coordinates for an array with the following axes:
        axis 0: variable
        axis 1: x-coordinate
        axis 2: y-coordinate
        axis 3: z-coordinate
    args:
        var (str) : variable name or tuple of variable names. if None, all variables are selected
        x (Tuple[int, int]) : x-coordinate slice. if None, all x-coordinates are selected
        y (Tuple[int, int]) : y-coordinate slice. if None, all y-coordinates are selected
        z (Tuple[int, int]) : z-coordinate slice. if None, all z-coordinates are selected
        ndim (int) : number of dimensions of the array
        axis (int) : axis to cut, alternative to x, y, z
        cut (Tuple[int, int]) : slice along dimension specified by axis. ignored if axis is None
        step (int) : step size for the slice. ignored if axis is None
    returns:
        tuple : slices for the given variable and coordinates
    """
    slices = [slice(None)] * ndim

    # variable slice
    if var is not None:
        if isinstance(var, str):
            if var not in VARIABLE_IDX_MAP:
                raise ValueError(f"Variable '{var}' not found.")
            slices[0] = VARIABLE_IDX_MAP[var]
        else:
            for v in var:
                if v not in VARIABLE_IDX_MAP:
                    raise ValueError(f"Variable '{v}' not found.")
            slices[0] = np.array(list(map(VARIABLE_IDX_MAP.get, var)))

    # x, y, z slices
    if x is not None:
        slices[1] = slice(x[0] or None, x[1] or None)
    if y is not None:
        slices[2] = slice(y[0] or None, y[1] or None)
    if z is not None:
        slices[3] = slice(z[0] or None, z[1] or None)

    # axis slice
    if axis is not None:
        if not (0 <= axis < ndim):
            raise ValueError(
                f"Axis {axis} is out of bounds for array with {ndim} dimensions."
            )
        if cut is not None:
            slices[axis] = slice(cut[0] or None, cut[1] or None, step)
    return tuple(slices)


class ArrayManager:
    """
    Class to manage arrays and their conversion between NumPy and CuPy.
    """

    def __init__(
        self,
    ):
        self.arrays = {}
        self.using_cupy = False

    def __repr__(self) -> str:
        return f"ArrayManager({self.arrays.keys()})"

    def enable_cupy(self):
        if CUPY_AVAILABLE:
            self.using_cupy = True
        else:
            print("WARNING: CuPy is not available. Falling back to NumPy.")

    def disable_cupy(self):
        self.using_cupy = False

    def add(self, name: str, array: np.ndarray):
        if name in self.arrays:
            raise KeyError(f"Array with name '{name}' already exists.")
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be of type numpy")
        if self.using_cupy:
            array = cp.asarray(array)
        self.arrays[name] = array

    def _check_name(self, name: str):
        if name not in self.arrays:
            raise KeyError(f"Array with name '{name}' not found.")

    def remove(self, name: str):
        self._check_name(name)
        del self.arrays[name]

    def get_numpy(self, name: str) -> np.ndarray:
        self._check_name(name)
        if self.using_cupy:
            return cp.asnumpy(self.arrays[name])
        return self.arrays[name].copy()

    def __call__(self, name: str) -> np.ndarray:
        self._check_name(name)
        return self.arrays[name]

    def to_dict(self) -> dict:
        return dict(names=list(self.arrays.keys()), using_cupy=self.using_cupy)
