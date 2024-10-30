from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from typing import Tuple, Union

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


@dataclass
class HydroState:
    """
    HydroState class for managing hydrodynamic variables and passive scalars.
        rho: density
        vx, vy, vz: velocity components
        mx, my, mz: momentum components
        P: pressure
        E: total energy
        your_passive_scalar1, your_passive_scalar2, ...: optional passive scalars

    attributes:
        passive_scalars (set) : set of passive scalar names
        ndim (int) : number of dimensions to slice by default
        variable_map (dict) : mapping of variable names to indices
        includes_passives (bool) : True if passive scalars are present
    """

    passive_scalars: tuple = ()
    ndim: int = 4

    def __init__(self, passive_scalars: tuple = (), ndim: int = 4):
        """
        args:
            passive_scalars (tuple) : tuple of passive scalar names. () if no passive scalars
            default_ndim (int) : number of dimensions to slice by default
        """
        self.passive_scalars = passive_scalars
        self.ndim = ndim
        self.variable_map = {
            "rho": 0,
            "vx": 1,
            "mx": 1,
            "vy": 2,
            "my": 2,
            "vz": 3,
            "mz": 3,
            "v": np.arange(1, 4),
            "m": np.arange(1, 4),
            "P": 4,
            "E": 4,
        }
        self.variable_map["active_scalars"] = np.array([0, 1, 2, 3, 4])

        # check no duplicate passive scalars
        if len(self.passive_scalars) != len(set(self.passive_scalars)):
            raise ValueError("Duplicate passive scalar names.")

        # add passive scalars to variable map
        if self.passive_scalars:
            for i, scalar in enumerate(self.passive_scalars, start=5):
                self.variable_map[scalar] = i
            self.variable_map["passive_scalars"] = np.arange(
                5, 5 + len(self.passive_scalars)
            )
            self.includes_passives = True
        else:
            self.includes_passives = False

        # set number of variables
        self.nvars = 5 + len(self.passive_scalars)

    def __hash__(self):
        return id(self)

    @lru_cache(maxsize=None)
    def __call__(
        self,
        var: Union[str, Tuple[str]] = None,
        x: Tuple[int, int] = None,
        y: Tuple[int, int] = None,
        z: Tuple[int, int] = None,
        axis: int = None,
        cut: Tuple[int, int] = None,
        step: int = None,
    ) -> Union[Tuple[slice], slice]:
        """
        Get the slice for the given variable and coordinates.
        args:
            var (str) : variable name or tuple of variable names. if None, all variables are selected
            x (Tuple[int, int]) : x-coordinate slice. if None, all x-coordinates are selected
            y (Tuple[int, int]) : y-coordinate slice. if None, all y-coordinates are selected
            z (Tuple[int, int]) : z-coordinate slice. if None, all z-coordinates are selected
            axis (int) : axis to cut, alternative to x, y, z
            cut (Tuple[int, int]) : slice along dimension specified by axis. ignored if axis is None
            step (int) : step size for the slice. ignored if axis is None
        returns:
            Tuple[slice] : slices for the given variable and coordinates with length equal to ndim.
                if ndim is 1, a single slice is returned
        """
        slices = [slice(None)] * self.ndim

        if var is not None:
            if isinstance(var, str):
                # retrieve single variable index
                if var not in self.variable_map:
                    raise ValueError(f"Variable '{var}' not found.")
                slices[0] = self.variable_map[var]
            elif isinstance(var, tuple):
                # retrieve multiple variable indices
                missing_vars = set(var) - set(self.variable_map.keys())
                if missing_vars:
                    raise ValueError(f"Variables not found: {missing_vars}")
                slices[0] = np.array(list(map(self.variable_map.get, var)))
            else:
                raise ValueError(f"Invalid type for var: {type(var)}")

        axes = [1, 2, 3, axis]
        axis_slices = [x, y, z, cut]
        for i, axis_slice in zip(axes, axis_slices):
            if axis_slice is None:
                continue
            if i >= self.ndim:
                raise ValueError(
                    f"Invalid axis {i} for array with {self.ndim} dimensions."
                )
            if not isinstance(axis_slice, tuple):
                raise ValueError(
                    f"Expected a tuple (start, stop) for axis {i}, got {axis_slice} of type {type(axis_slice)}"
                )
            if len(axis_slice) != 2:
                raise ValueError(
                    f"Invalid tuple length for axis {i}: {len(axis_slice)}"
                )
            slices[i] = slice(
                axis_slice[0] or None,
                axis_slice[1] or None,
                step if i == axis else None,
            )

        if len(slices) == 1:
            return slices[0]
        return tuple(slices)


VARIABLE_IDX_MAP = {
    "rho": 0,
    "P": 4,
    "E": 4,
    "vx": 1,
    "mx": 1,
    "vy": 2,
    "my": 2,
    "vz": 3,
    "mz": 3,
}


@lru_cache(maxsize=None)
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
