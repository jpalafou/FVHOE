from dataclasses import dataclass, field
from itertools import product
from fvhoe.array_manager import ArrayManager, get_array_slice as slc
import numpy as np
from typing import Dict, Tuple, Union

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


def set_dirichlet_bc(
    u: np.ndarray,
    boundary_values: Union[np.ndarray, callable],
    slab_thickness: int,
    axis: int,
    pos: str,
    **kwargs,
) -> None:
    """
    modify u to impose dirichlet boundaries with either constant values or a function
    args:
        u (array_like) : array to apply boundary conditions to
        boundary_values (array_like, callable) : 1D array or callable that returns 3D array
        slab_thickness (int) : thickness of the slab to be copied
        axis (int) : axis to apply periodic boundary conditions
        pos (str) : position of the slab to be copied, "l" or "r"
    returns:
        None, modifies u in place
    """
    slab_view = u[
        slc(
            axis=axis,
            cut={"l": (0, slab_thickness), "r": (-slab_thickness, 0)}[pos],
        )
    ]

    if callable(boundary_values):
        dirichlet_values = boundary_values(**kwargs)
    elif isinstance(boundary_values, np.ndarray) or isinstance(
        boundary_values, cp.ndarray
    ):
        dirichlet_values = np.ones_like(slab_view) * boundary_values.reshape(
            -1, 1, 1, 1
        )
    else:
        raise ValueError(f"Invalid boundary of type '{boundary_values}'.")

    slab_view[...] = dirichlet_values


def set_free_bc(u: np.ndarray, slab_thickness: int, axis: int, pos: str):
    """
    modify u to impose free boundaries
    args:
        u: (array_like) array to apply boundary conditions to
        slab_thickness: (int) thickness of the slab to be copied
        axis: (int) axis to apply periodic boundary conditions
        pos: (str) position of the slab to be copied, "l" or "r"
    returns:
        None, modifies u in place
    """
    st = slab_thickness

    if pos == "l":
        outer = slc(axis=axis, cut=(0, st))
        inner = slc(axis=axis, cut=(st, (st + 1)))
    elif pos == "r":
        outer = slc(axis=axis, cut=(-st, 0))
        inner = slc(axis=axis, cut=(-(st + 1), -st))
    else:
        raise ValueError(f"Invalid pos '{pos}'")

    u[outer] = u[inner]


def set_periodic_bc(u: np.ndarray, slab_thickness: int, axis: int) -> None:
    """
    modify u to impose periodic boundaries
    args:
        u: (array_like) array to apply periodic boundary conditions to
        slab_thickness: (int) thickness of the slab to be copied
        axis: (int) axis to apply periodic boundary conditions
    returns:
        None, modifies u in place
    """
    st = slab_thickness
    outer_l = slc(axis=axis, cut=(0, st))
    inner_l = slc(axis=axis, cut=(st, 2 * st))
    outer_r = slc(axis=axis, cut=(-st, 0))
    inner_r = slc(axis=axis, cut=(-2 * st, -st))
    u[outer_l] = u[inner_r]
    u[outer_r] = u[inner_l]


def set_symmetric_bc(
    u: np.ndarray,
    slab_thickness: int,
    axis: int,
    pos: str,
    negate_var: str = None,
) -> None:
    """
    modify u to impose symmetric boundaries
    args:
        u: (array_like) array to apply boundary conditions to
        slab_thickness: (int) thickness of the slab to be copied
        axis: (int) axis to apply periodic boundary conditions
        pos: (str) position of the slab to be copied, "l" or "r"
        negate_var (str) : optional. variable to multiply by -1 in the slab
    returns:
        None, modifies u in place
    """
    st = slab_thickness
    flip = slc(axis=axis, cut=(0, 0), step=-1)
    if pos == "l":
        outer = slc(axis=axis, cut=(0, st))
        inner = slc(axis=axis, cut=(st, 2 * st))
    elif pos == "r":
        outer = slc(axis=axis, cut=(-st, 0))
        inner = slc(axis=axis, cut=(-2 * st, -st))
    else:
        raise ValueError(f"Invalid pos '{pos}'")

    u[outer] = u[inner][flip]

    # multiply a variable in the slab by -1
    if negate_var is not None:
        u[
            slc(
                negate_var,
                axis=axis,
                cut={"l": (0, st), "r": (-st, 0)}[pos],
            )
        ] *= -1.0


@dataclass
class BoundaryCondition:
    """
    Specify and apply various boundary conditions in 3D
    x (Union[str, Tuple[str, str]]) : boundary conditions in x-direction
        tuple : left bc type, right bc type
        single value : both bc types
    y (Union[str, Tuple[str, str]]) : boundary conditions in y-direction
    z (Union[str, Tuple[str, str]]) : boundary conditions in z-direction
    x_value (Union[Any, Tuple[Any, Any]]) : data for dirichlet boundaries in x-direction
        tuple : left bc value, right bc value
        single value : both bc values
    y_value (Union[Any, Tuple[Any, Any]]) : data for dirichlet boundaries in y-direction
    z_value (Union[Any, Tuple[Any, Any]]) : data for dirichlet boundaries in z-direction
    slab_coords (Dict[str, np.ndarray]) : dict of coordinates meshes of slabs in 3D
        {"xl": (X, Y, Z), ...}
    array_manager (ArrayManager) : array manager to allocate dirichlet arrays
    """

    x: Union[str, Tuple[str, str]] = "periodic"
    y: Union[str, Tuple[str, str]] = "periodic"
    z: Union[str, Tuple[str, str]] = "periodic"
    x_value: Union[
        Union[callable, np.ndarray],
        Tuple[Union[callable, np.ndarray], Union[callable, np.ndarray]],
    ] = None
    y_value: Union[
        Union[callable, np.ndarray],
        Tuple[Union[callable, np.ndarray], Union[callable, np.ndarray]],
    ] = None
    z_value: Union[
        Union[callable, np.ndarray],
        Tuple[Union[callable, np.ndarray], Union[callable, np.ndarray]],
    ] = None
    slab_coords: Dict[str, np.ndarray] = None
    array_manager: ArrayManager = field(default_factory=ArrayManager)

    def __post_init__(self):
        # convert single values to tuples
        for dim, suffix in product("xyz", ["", "_value"]):
            value = getattr(self, f"{dim}{suffix}")
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f"Invalid length of {dim}{suffix} tuple")
            elif isinstance(value, list):
                raise ValueError(f"Invalid type of {dim}{suffix}: list")
            else:
                setattr(self, f"{dim}{suffix}", (value, value))

        # allocate dirichlet arrays
        for dim in "xyz":
            l_value, r_value = getattr(self, f"{dim}_value")
            if isinstance(l_value, np.ndarray):
                self.array_manager.add(f"bc_{dim}l_value", l_value)
                self.reset_value(dim, "l", self.array_manager(f"bc_{dim}l_value"))
            if isinstance(r_value, np.ndarray):
                self.array_manager.add(f"bc_{dim}r_value", r_value)
                self.reset_value(dim, "r", self.array_manager(f"bc_{dim}r_value"))

        # allocate slab coords
        if self.slab_coords is None:
            self.slab_buffer_size = (0, 0, 0)
        else:
            self.slab_buffer_size = (
                self.slab_coords["xl"][0].shape[0],
                self.slab_coords["yl"][0].shape[1],
                self.slab_coords["zl"][0].shape[2],
            )
            for dim in "xyz":
                for pos in "lr":
                    X, Y, Z = self.slab_coords[dim + pos]
                    self.array_manager.add(f"bc_{dim}{pos}_slab_x", X)
                    self.array_manager.add(f"bc_{dim}{pos}_slab_y", Y)
                    self.array_manager.add(f"bc_{dim}{pos}_slab_z", Z)

    def reset_value(self, dim: str, pos: str, new_value):
        """
        reset the first or second element of self.x_value, self.y_value or self.z_value
        args:
            dim (str) : dimension to reset ("x", "y", or "z")
            pos (str) : position to reset ("l" or "r")
            new_value (array_like) : new value to set
        """
        l_value, r_value = getattr(self, f"{dim}_value")
        if pos == "l":
            l_value = new_value
        elif pos == "r":
            r_value = new_value
        else:
            raise ValueError(f"Invalid pos '{pos}'")
        setattr(self, f"{dim}_value", (l_value, r_value))

    def trim_slabs(
        self, gw: Tuple[int, int, int], axis: int, pos: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        View useful region of pre-allocated slab coordinates
        args:
            gw (Tuple[int, int, int]) : ghost zone width in x, y, and z
            axis (int) : axis of desired slab
            pos (str) : slab position along axis ("l" or "r")
        """
        X = self.array_manager(f"bc_{'xyz'[axis]}{pos}_slab_x")
        Y = self.array_manager(f"bc_{'xyz'[axis]}{pos}_slab_y")
        Z = self.array_manager(f"bc_{'xyz'[axis]}{pos}_slab_z")
        xexcess = self.slab_buffer_size[0] - gw[0]
        yexcess = self.slab_buffer_size[1] - gw[1]
        zexcess = self.slab_buffer_size[2] - gw[2]
        if xexcess < 0:
            raise ValueError(
                f"Cannot apply bc to region of thickness {gw[0]} with a buffer of thickness {self.slab_buffer_size[0]}"
            )
        if yexcess < 0:
            raise ValueError(
                f"Cannot apply bc to region of thickness {gw[1]} with a buffer of thickness {self.slab_buffer_size[1]}"
            )
        if zexcess < 0:
            raise ValueError(
                f"Cannot apply bc to region of thickness {gw[2]} with a buffer of thickness {self.slab_buffer_size[2]}"
            )
        xtrim = slc(
            ndim=3,
            axis=0,
            cut=(
                {"l": (-gw[0], 0), "r": (0, gw[0])}[pos]
                if axis == 0
                else (xexcess, -xexcess)
            ),
        )
        ytrim = slc(
            ndim=3,
            axis=1,
            cut=(
                {"l": (-gw[1], 0), "r": (0, gw[1])}[pos]
                if axis == 1
                else (yexcess, -yexcess)
            ),
        )
        ztrim = slc(
            ndim=3,
            axis=2,
            cut=(
                {"l": (-gw[2], 0), "r": (0, gw[2])}[pos]
                if axis == 2
                else (zexcess, -zexcess)
            ),
        )
        X = X[xtrim][ytrim][ztrim]
        Y = Y[xtrim][ytrim][ztrim]
        Z = Z[xtrim][ytrim][ztrim]
        return X, Y, Z

    def apply(self, u: np.ndarray, gw: Tuple[int, int, int], t: float = None):
        """
        Apply boundary conditions to conservative variable array u, increasing its size.
        args:
            u (array_like) : array of conservative variables in 3D
            gw (Tuple[int, int, int]) : ghost zone width in x, y, and z
            t : (float) time value
        returns:
            out (array_like) : arrays padded according to gw
        """
        # apply temporary boundaries of nan
        out = np.pad(
            u,
            pad_width=[(0, 0), (gw[0], gw[0]), (gw[1], gw[1]), (gw[2], gw[2])],
            mode="empty",
        )

        # loop through slabs
        for (i, dim), (j, pos) in product(enumerate("xyz"), enumerate("lr")):
            # gather bc parameters
            bc = getattr(self, dim)[j]
            slab_thickness = gw[i]

            # skip if no thickness
            if slab_thickness == 0:
                continue

            match bc:
                case "dirichlet" | "ic":
                    # set dirichlet boundaries
                    X, Y, Z = self.trim_slabs(gw=gw, axis=i, pos=pos)
                    dirichlet_kwargs = dict(x=X, y=Y, z=Z)
                    if bc == "dirichlet":
                        dirichlet_kwargs["t"] = t
                    set_dirichlet_bc(
                        u=out,
                        boundary_values=getattr(self, f"{dim}_value")[j],
                        slab_thickness=slab_thickness,
                        axis=i + 1,
                        pos=pos,
                        **dirichlet_kwargs,
                    )
                case "free":
                    set_free_bc(out, slab_thickness, axis=i + 1, pos=pos)
                case "outflow" | "reflective":
                    set_symmetric_bc(
                        out,
                        slab_thickness,
                        axis=i + 1,
                        pos=pos,
                        negate_var=f"m{dim}" if bc == "reflective" else None,
                    )
                case "periodic":
                    if pos == "l":  # do not apply periodic bc's twice
                        set_periodic_bc(out, slab_thickness, axis=i + 1)
                case "special-case-double-mach-reflection-y=0":
                    X = self.trim_slabs(gw=gw, axis=1, pos="l")[0][
                        np.newaxis, :, :1, :1
                    ]
                    out_free = out.copy()
                    out_refl = out.copy()

                    # set free bc
                    set_free_bc(out_free, slab_thickness, axis=i + 1, pos=pos)

                    # set reflective bc
                    set_symmetric_bc(
                        out_refl,
                        slab_thickness,
                        axis=i + 1,
                        pos=pos,
                        negate_var="my",
                    )

                    # piecewise combination of the two
                    out = np.where(X < 1 / 6, out_free, out_refl)

                case None:
                    pass
                case _:
                    raise ValueError(f"Unknown bc type '{bc}'")

        return out

    def to_dict(self) -> dict:
        """
        return JSON-able dict
        """
        return dict(
            x=self.x,
            y=self.y,
            z=self.z,
        )
