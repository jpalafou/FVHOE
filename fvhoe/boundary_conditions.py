from dataclasses import dataclass
from itertools import product
from functools import partial
from fvhoe.config import conservative_names
from fvhoe.fv import get_view
from fvhoe.named_array import NamedNumpyArray, NamedCupyArray
import numpy as np
from typing import Any, Dict, Tuple, Union


def set_dirichlet_bc(
    u: NamedNumpyArray,
    boundary_values: Union[NamedNumpyArray, callable],
    slab_thickness: int,
    axis: int,
    pos: str,
    **kwargs,
) -> None:
    """
    args:
        u (NamedArray) : array to apply boundary conditions to
        boundary_values (NamedArray, callable) : 1D NamedArray or callable that returns 3D array
        slab_thickness (int) : thickness of the slab to be copied
        axis (int) : axis to apply periodic boundary conditions
        pos (str) : position of the slab to be copied, "l" or "r"
    returns:
        None, modifies u in place
    """
    slab_view = get_view(
        ndim=4,
        axis=axis,
        cut={"l": (0, -slab_thickness), "r": (-slab_thickness, 0)}[pos],
    )

    if callable(boundary_values):
        dirichlet_values = boundary_values(**kwargs)
    elif isinstance(boundary_values, NamedNumpyArray) or isinstance(
        boundary_values, NamedCupyArray
    ):
        dirichlet_values = np.ones_like(u[slab_view]) * boundary_values.reshape(
            -1, 1, 1, 1
        )
    else:
        raise ValueError(f"Invalid boundary of type '{boundary_values}'.")

    u[slab_view] = dirichlet_values


def set_free_bc(u: NamedNumpyArray, slab_thickness: int, axis: int, pos: str):
    """
    modify u to impose free boundaries
    args:
        u: (NamedArray) array to apply boundary conditions to
        slab_thickness: (int) thickness of the slab to be copied
        axis: (int) axis to apply periodic boundary conditions
        pos: (str) position of the slab to be copied, "l" or "r"
    returns:
        None, modifies u in place
    """
    gv = partial(get_view, ndim=4, axis=axis)
    st = slab_thickness

    if pos == "l":
        u[gv(cut=(0, -st))] = u[gv(cut=(st, -(st + 1)))].copy()
    elif pos == "r":
        u[gv(cut=(-st, 0))] = u[gv(cut=(-(st + 1), st))].copy()
    else:
        raise ValueError(f"Invalid pos '{pos}'")


def set_periodic_bc(u: NamedNumpyArray, slab_thickness: int, axis: int) -> None:
    """
    modify u to impose periodic boundary conditions
    args:
        u: (NamedArray) array to apply periodic boundary conditions to
        slab_thickness: (int) thickness of the slab to be copied
        axis: (int) axis to apply periodic boundary conditions
    returns:
        None, modifies u in place
    """
    pad_width = [(0, 0), (0, 0), (0, 0), (0, 0)]
    pad_width[axis] = (slab_thickness, slab_thickness)
    u[...] = np.pad(
        u[get_view(ndim=4, axis=axis, cut=(slab_thickness, slab_thickness))],
        pad_width=pad_width,
        mode="wrap",
    )


def set_symmetric_bc(
    u: NamedNumpyArray, slab_thickness: int, axis: int, pos: str
) -> None:
    """
    modify u to impose symmetric boundaries
    args:
        u: (NamedArray) array to apply boundary conditions to
        slab_thickness: (int) thickness of the slab to be copied
        axis: (int) axis to apply periodic boundary conditions
        pos: (str) position of the slab to be copied, "l" or "r"
    returns:
        None, modifies u in place
    """
    gv = partial(get_view, ndim=4, axis=axis)
    st = slab_thickness

    if pos == "l":
        u[gv(cut=(0, -st))] = u[gv(cut=(st, -2 * st))][gv(step=-1)].copy()
    elif pos == "r":
        u[gv(cut=(-st, 0))] = u[gv(cut=(-2 * st, st))][gv(step=-1)].copy()
    else:
        raise ValueError(f"Invalid pos '{pos}'")


@dataclass
class BoundaryCondition:
    names: list = None
    x: Union[str, Tuple[str, str]] = "periodic"
    y: Union[str, Tuple[str, str]] = "periodic"
    z: Union[str, Tuple[str, str]] = "periodic"
    x_value: Tuple[Any, Any] = None
    y_value: Tuple[Any, Any] = None
    z_value: Tuple[Any, Any] = None
    slab_coords: Dict[str, np.ndarray] = None

    def __post_init__(self):
        # ensure all bcs and bc values are tuples
        for dim in "xyz":
            bc = getattr(self, dim)
            bc_value = getattr(self, f"{dim}_value")
            if not (isinstance(bc, tuple) or isinstance(bc, list)):
                setattr(self, dim, (bc, bc))
            if not (isinstance(bc_value, tuple) or isinstance(bc_value, list)):
                setattr(self, f"{dim}_value", (bc_value, bc_value))

        # get slab buffer sizes
        if self.slab_coords is None:
            self.slab_buffer_size = (0, 0, 0)
        else:
            self.slab_buffer_size = (
                self.slab_coords["xl"][0].shape[0],
                self.slab_coords["xl"][0].shape[1],
                self.slab_coords["xl"][0].shape[2],
            )

    def apply(self, u: NamedNumpyArray, gw: Tuple[int, int, int], t: float = None):
        # apply temporary boundaries of nan
        out = np.pad(
            u,
            pad_width=[(0, 0), (gw[0], gw[0]), (gw[1], gw[1]), (gw[2], gw[2])],
            mode="empty",
        )
        out = u.__class__(input_array=out, names=conservative_names)

        for (i, dim), (j, pos) in product(enumerate("xyz"), enumerate("lr")):
            bc = getattr(self, dim)[j]
            slab_thickness = gw[i]
            if slab_thickness == 0:
                continue

            match bc:
                case "dirichlet" | "ic":
                    # get slab coordinates
                    slab_coords = self.slab_coords[dim + pos]
                    # trim excess buffer
                    if self.slab_buffer_size[i] < slab_thickness:
                        raise BaseException(
                            f"Slab thickness {self.slab_buffer_size[i]} too small to apply {slab_thickness} boundaries along {dim}-dimension."
                        )
                    trim = get_view(
                        ndim=3,
                        axis=i,
                        cut={"l": (-slab_thickness, 0), "r": (0, -slab_thickness)}[pos],
                    )
                    # set dirichlet boundaries
                    dirichlet_kwargs = dict(
                        x=slab_coords[0][trim],
                        y=slab_coords[1][trim],
                        z=slab_coords[2][trim],
                    )
                    if bc == "dirichlet":
                        dirichlet_kwargs["t"] = t
                    set_dirichlet_bc(
                        out,
                        getattr(self, f"{dim}_value")[j],
                        slab_thickness,
                        axis=i + 1,
                        pos=pos,
                        **dirichlet_kwargs,
                    )
                case "free":
                    set_free_bc(out, slab_thickness, axis=i + 1, pos=pos)
                case "outflow" | "reflective":
                    set_symmetric_bc(out, slab_thickness, axis=i + 1, pos=pos)
                    if bc == "reflective":  # multiply momentum at boundaries by -1
                        getattr(out, f"m{dim}")[
                            get_view(
                                ndim=3,
                                axis=i,
                                cut={
                                    "l": (0, -slab_thickness),
                                    "r": (-slab_thickness, 0),
                                }[pos],
                            )
                        ] *= -1
                case "periodic":
                    if pos == "l":  # do not apply periodic bc's twice
                        set_periodic_bc(out, slab_thickness, axis=i + 1)
                case None:
                    pass
                case _:
                    raise ValueError(f"Unknown bc type '{bc}'")

        return out
