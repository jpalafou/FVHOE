from dataclasses import dataclass, field
import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


@dataclass
class ArrayManager:
    """
    Class to manage arrays and their conversion between NumPy and CuPy.
    """

    arrays: dict = field(default_factory=dict)
    using_cupy: bool = False

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
