from dataclasses import dataclass
from itertools import product
from fvhoe.array_manager import get_array_slice as slc
from fvhoe.fv import get_view
import numpy as np
from typing import Any, Dict, Tuple, Union

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
        boundary_values (array_like, callable) : 1D NamedArray or callable that returns 3D array
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
    args:
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
        cupy (bool) : whether to use cupy
    """

    x: Union[str, Tuple[str, str]] = "periodic"
    y: Union[str, Tuple[str, str]] = "periodic"
    z: Union[str, Tuple[str, str]] = "periodic"
    x_value: Union[Any, Tuple[Any, Any]] = None
    y_value: Union[Any, Tuple[Any, Any]] = None
    z_value: Union[Any, Tuple[Any, Any]] = None
    slab_coords: Dict[str, np.ndarray] = None
    cupy: bool = False

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
                self.slab_coords["yl"][0].shape[1],
                self.slab_coords["zl"][0].shape[2],
            )

        # convert necessary arrays to cupy
        if self.cupy and CUPY_AVAILABLE:
            # dirichlet value arrays
            for dim in "xyz":
                bc_value = list(getattr(self, f"{dim}_value"))
                for i in [0, 1]:
                    if isinstance(bc_value[i], np.ndarray):
                        bc_value[i] = cp.asarray(bc_value[i])
                setattr(self, f"{dim}_value", tuple(bc_value))

            # slab coords
            for dim in "xyz":
                for pos in "lr":
                    x, y, z = self.slab_coords[dim + pos]
                    x, y, z = cp.asarray(x), cp.asarray(y), cp.asarray(z)
                    self.slab_coords[dim + pos] = x, y, z

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
        X, Y, Z = self.slab_coords[{0: "x", 1: "y", 2: "z"}[axis] + pos]
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
        xtrim = get_view(
            ndim=3,
            axis=0,
            cut=(
                {"l": (-gw[0], 0), "r": (0, -gw[0])}[pos]
                if axis == 0
                else (xexcess, xexcess)
            ),
        )
        ytrim = get_view(
            ndim=3,
            axis=1,
            cut=(
                {"l": (-gw[1], 0), "r": (0, -gw[1])}[pos]
                if axis == 1
                else (yexcess, yexcess)
            ),
        )
        ztrim = get_view(
            ndim=3,
            axis=2,
            cut=(
                {"l": (-gw[2], 0), "r": (0, -gw[2])}[pos]
                if axis == 2
                else (zexcess, zexcess)
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
            cupy=self.cupy,
        )
