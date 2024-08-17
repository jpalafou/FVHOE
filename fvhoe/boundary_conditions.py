from dataclasses import dataclass
from itertools import product
from functools import partial
from fvhoe.config import conservative_names
from fvhoe.fv import get_view
from fvhoe.named_array import NamedNumpyArray, NamedCupyArray
import numpy as np
from typing import Any, Dict, Tuple, Union

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


def set_dirichlet_bc(
    u: NamedNumpyArray,
    boundary_values: Union[NamedNumpyArray, callable],
    slab_thickness: int,
    axis: int,
    pos: str,
    **kwargs,
) -> None:
    """
    modify u to impose dirichlet boundaries with either constant values or a function
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
    modify u to impose periodic boundaries
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
    u: NamedNumpyArray,
    slab_thickness: int,
    axis: int,
    pos: str,
    negate_var: str = None,
) -> None:
    """
    modify u to impose symmetric boundaries
    args:
        u: (NamedArray) array to apply boundary conditions to
        slab_thickness: (int) thickness of the slab to be copied
        axis: (int) axis to apply periodic boundary conditions
        pos: (str) position of the slab to be copied, "l" or "r"
        negate_var (str) : optional. variable to multiply by -1 in the slab
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

    # multiply a variable in the slab by -1
    if negate_var is not None:
        var_data = getattr(u, negate_var)
        var_data[
            get_view(
                ndim=3,
                axis=axis - 1,
                cut={
                    "l": (0, -slab_thickness),
                    "r": (-slab_thickness, 0),
                }[pos],
            )
        ] *= -1.0
        setattr(u, negate_var, var_data)


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
                    if isinstance(bc_value[i], NamedNumpyArray):
                        bc_value[i] = NamedCupyArray(
                            input_array=bc_value[i],
                            names=bc_value[i].variable_names,
                        )
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

    def apply(self, u: NamedNumpyArray, gw: Tuple[int, int, int], t: float = None):
        """
        Apply boundary conditions to conservative variable array u, increasing its size.
        args:
            u (NamedArray) : array of conservative variables in 3D
            gw (Tuple[int, int, int]) : ghost zone width in x, y, and z
            t : (float) time value
        returns:
            out (NamedArray) : arrays padded according to gw
        """
        # apply temporary boundaries of nan
        out = np.pad(
            u,
            pad_width=[(0, 0), (gw[0], gw[0]), (gw[1], gw[1]), (gw[2], gw[2])],
            mode="empty",
        )
        out = u.__class__(input_array=out, names=conservative_names)

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
                    out = u.__class__(input_array=out, names=conservative_names)

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
