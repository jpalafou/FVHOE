import numpy as np
from typing import Iterable, Union

try:
    import cupy as cp

    _ = cp.asarray(np.array([1, 2]))
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


def defined_NamedArray_class(cupy: bool = False):
    """
    define NamedArray class using numpy or cupy
    args:
        cupy (bool) : whether to use cupy library
    returns:
        NamedArray class
    """

    # define xp as either numpy or cupy
    xp = np
    xp_asnumpy = np.asarray
    if cupy and CUPY_AVAILABLE:
        xp = cp
        xp_asnumpy = cp.asnumpy
    elif cupy:
        print("WARNING: CuPy is not available. Falling back to NumPy.")

    # define NamedArray class
    class NamedArray(xp.ndarray):
        def __new__(cls, input_array: xp.ndarray, names: Iterable, copy: bool = False):
            """
            array with first axis indexable by specified names
            args:
                input_array (array_like) : array of shape (len(names), ...)
                names (Iterable) : series of names, must be valid python variable names
                copy (bool) : True -> returns a copy of the data, False -> returns a view
            """
            if input_array.shape[0] != len(names):
                raise TypeError(
                    "input_array first axis length does not match length of names"
                )

            obj = xp.array(input_array, copy=copy).view(cls)

            # Directly set the attributes to avoid calling __setattr__
            obj.__dict__["variable_indices"] = {name: i for i, name in enumerate(names)}
            obj.__dict__["variable_names"] = names
            obj.__dict__["variable_name_set"] = set(names)
            obj.__dict__["xp"] = xp.__name__
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self.variable_indices = getattr(obj, "variable_indices", None)
            self.variable_names = getattr(obj, "variable_names", None)
            self.variable_name_set = getattr(obj, "variable_name_set", set())
            self.xp = getattr(obj, "xp", None)

        def __getattr__(self, name: str) -> xp.ndarray:
            """
            Return slice of array as numpy or cupy array.
            """
            if name in self.variable_name_set:
                return xp.array(
                    self[self.variable_indices[name], ...], copy=self.xp == "cupy"
                )
            else:
                raise AttributeError(f"'NamedArray' object has no attribute '{name}'")

        def __setattr__(self, name, value):
            if "variable_name_set" in self.__dict__ and name in self.variable_name_set:
                self[self.variable_indices[name], ...] = value
            else:
                super().__setattr__(name, value)

        def copy(self):
            return self.__class__(self, self.variable_names, copy=True)

        def asnumpy(self) -> np.ndarray:
            """
            Return array as numpy array.
            """
            return xp_asnumpy(self)

        def rename_variables(self, rename: Union[dict, Iterable]):
            """
            rename variables
            args:
                new_names (dict or Iterable) : map from old names to new names, or series of new names
            """
            if isinstance(rename, dict):
                rename_dict = {x: rename.get(x, x) for x in self.variable_names}
                new_names = list(rename_dict.values())
            else:
                new_names = rename
                if self.shape[0] != len(new_names):
                    raise TypeError(
                        "input_array first axis length does not match length of new_names"
                    )
            out = self.__class__(self, new_names, copy=True)
            return out

        def remove(self, name: Union[str, Iterable]):
            """
            remove variables
            args:
                name (str or Iterable) : names of variables to remove
            """
            names = [name] if isinstance(name, str) else name

            # delete rows from first axis
            remove_idxs = tuple(self.variable_indices[iname] for iname in names)
            out = np.delete(self, remove_idxs, axis=0)

            # revise variable data
            for iname in names:
                out.variable_names = list(out.variable_names)
                out.variable_names.remove(iname)  # remove iname from list
                out.variable_name_set -= set([iname])  # remove iname from set
            out.variable_indices = {
                namei: i for i, namei in enumerate(out.variable_names)
            }

            return out

        def merge(self, other, copy=False):
            """
            merge NamedArray with another NamedArray
            """
            # find and remove_common variables
            common_variables = list(self.variable_name_set & other.variable_name_set)
            if len(common_variables) > 0:
                other = other.remove(common_variables)

            # concatenate arrays
            out_arr = np.concatenate((self, other), axis=0)
            combined_variable_names = list(self.variable_names) + list(
                other.variable_names
            )
            out = self.__class__(out_arr, combined_variable_names, copy)

            return out

    return NamedArray


NamedNumpyArray = defined_NamedArray_class(cupy=False)
NamedCupyArray = defined_NamedArray_class(cupy=True)


def asnamednumpy(self):
    """
    Convert named CuPy array to named NumPy array
    """
    return NamedNumpyArray(self.asnumpy(), self.variable_names)


NamedCupyArray.asnamednumpy = asnamednumpy
