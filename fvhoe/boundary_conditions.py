from dataclasses import dataclass
from itertools import product
from functools import partial
from fvhoe.fv import get_view, uniform_fv_mesh
from fvhoe.config import conservative_names
from fvhoe.named_array import NamedNumpyArray
import numpy as np
from typing import Iterable, Tuple

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


def set_dirichlet_bc(
    f: callable,
    u: np.ndarray,
    num_ghost: int,
    dim: str,
    pos: str,
    x: np.ndarray = None,
    y: np.ndarray = None,
    z: np.ndarray = None,
    t: float = None,
) -> None:
    """
    set Dirichlet boundaries
    args:
        f (callable, float) : defines boundary values
            f(x, y, z, t) (callable) : boundary function
            f (float) : uniform boundary
        u (array_like) : padded array of shape (nx, ny, nz)
        num_ghost (int) : number of 'ghost zones' on pos end of domain
        dim (str) : dimension
            "x" : axis = 0
            "y" : axis = 1
            "z" : axis = 2
        pos (str) : left or right boundary of selected dimension
            "l" : left
            "r" : right
        x (array_like) : mesh of x-values, has shape (nx, ny, nz)
        y (array_like) : mesh of y-values, has shape (nx, ny, nz)
        z (array_like) : mesh of z-values, has shape (nx, ny, nz)
    returns:
        u : (array_like) : u with boundary conditions applied
    """
    gv = partial(get_view, ndim=u.ndim, axis={"x": 0, "y": 1, "z": 2}[dim])
    x = np.array([]) if None else x
    y = np.array([]) if None else y
    z = np.array([]) if None else z

    # if setting left side boundary: reflect, compute right boundary, reflect again
    if pos == "l":
        u[...] = u[gv(step=-1)]
        x, y, z = x[gv(step=-1)], y[gv(step=-1)], z[gv(step=-1)]
        set_dirichlet_bc(
            f=f,
            u=u,
            num_ghost=num_ghost,
            dim=dim,
            pos="r",
            x=x,
            y=y,
            z=z,
            t=t,
        )
        u[...] = u[gv(step=-1)]
        x, y, z = x[gv(step=-1)], y[gv(step=-1)], z[gv(step=-1)]
        return u

    # set right side boundary
    if num_ghost > 0:
        boundary_view = gv(cut=(-num_ghost, 0))
        if callable(f):
            x_bc, y_bc, z_bc = x[boundary_view], y[boundary_view], z[boundary_view]
            boundary_values = f(x_bc, y_bc, z_bc, t)
        else:
            boundary_values = f
        u[boundary_view] = boundary_values
    return u


def set_periodic_bc(u: np.ndarray, num_ghost: int, dim: str) -> None:
    """
    set boundaries using np pad
    args:
        u (array_like) : padded array
        num_ghost (int) : number of 'ghost zones' on pos end of domain
        dim (str) : dimension
            "x" : axis = 0
            "y" : axis = 1
            "z" : axis = 2
    returns:
        None : revises u
    """
    axis = {"x": 0, "y": 1, "z": 2}[dim]
    gv = partial(get_view, ndim=u.ndim, axis=axis)
    pad_width = [(0, 0), (0, 0), (0, 0)]
    pad_width[axis] = (num_ghost, num_ghost)
    u[...] = np.pad(u[gv(cut=(num_ghost, num_ghost))], pad_width=pad_width, mode="wrap")
    return u


def set_symmetric_bc(
    u: np.ndarray, num_ghost: int, dim: str, pos: str, negative: bool = False
) -> None:
    """
    set reflective boundaries
    args:
        u (array_like) : padded array
        num_ghost (int) : number of 'ghost zones' on  pos end of domain
        dim (str) : dimension
            "x" : axis = 0
            "y" : axis = 1
            "z" : axis = 2
        pos (str) : left or right boundary of selected dimension
            "l" : left
            "r" : right
        negative (bool) : whether to multiply boundary values by -1
    returns:
        u : (array_like) : u with boundary conditions applied
    """
    axis = {"x": 0, "y": 1, "z": 2}[dim]
    gv = partial(get_view, ndim=u.ndim, axis=axis)
    pad_width = [(0, 0), (0, 0), (0, 0)]
    if pos == "l":
        u[...] = u[gv(step=-1)]
        set_symmetric_bc(u=u, num_ghost=num_ghost, dim=dim, pos="r", negative=negative)
        u[...] = u[gv(step=-1)]
        return u

    pad_width[axis] = (0, num_ghost)
    u[...] = np.pad(u[gv(cut=(0, num_ghost))], pad_width=pad_width, mode="symmetric")
    if negative:
        u[gv(cut=(-num_ghost, 0))] *= -1

    return u


def set_free_bc(u: np.ndarray, num_ghost: int, dim: str, pos: str) -> None:
    """
    set free boundaries
    args:
        u (array_like) : padded array
        num_ghost (int) : number of 'ghost zones' on  pos end of domain
        dim (str) : dimension
            "x" : axis = 0
            "y" : axis = 1
            "z" : axis = 2
        pos (str) : left or right boundary of selected dimension
            "l" : left
            "r" : right
    returns:
        u : (array_like) : u with boundary conditions applied
    """
    axis = {"x": 0, "y": 1, "z": 2}[dim]
    gv = partial(get_view, ndim=u.ndim, axis=axis)
    pad_width = [(0, 0), (0, 0), (0, 0)]
    if pos == "l":
        u[...] = u[gv(step=-1)]
        set_free_bc(u=u, num_ghost=num_ghost, dim=dim, pos="r")
        u[...] = u[gv(step=-1)]
        return u

    pad_width[axis] = (0, num_ghost)
    u[...] = np.pad(u[gv(cut=(0, num_ghost))], pad_width=pad_width, mode="edge")
    return u


def fd(
    u: np.ndarray,
    p: int,
    h: float,
    axis: int,
) -> np.ndarray:
    """
    compute finite difference of u
    args:
        u (array_like) : array of values
        p (int) : interpolation polynomial degree
        h (float) : grid spacing along axis
        axis (int) : along which to apply bcs
    returns:
        out : finite difference approximations
    """

    gv = partial(get_view, ndim=u.ndim, axis=axis)

    if p == 0:
        out = u.copy()
    elif p in (1, 2):
        out = (1 / (2 * h)) * (1 * u[gv(cut=(2, 0))] + -1 * u[gv(cut=(0, 2))])
    elif p in (3, 4):
        out = (1 / (12 * h)) * (
            -1 * u[gv(cut=(4, 0))]
            + 8 * u[gv(cut=(3, 1))]
            + -8 * u[gv(cut=(1, 3))]
            + 1 * u[gv(cut=(0, 4))]
        )
    else:
        raise NotImplementedError(f"{p=}")
    return out


VALID_BC_TYPES = {
    None,
    "dirichlet",
    "free",
    "periodic",
    "special-case-double-mach-reflection-y=0",
    "symmetric",
    "-symmetric",
}


@dataclass
class BoundaryCondition:
    """
    Boundary condition class for a NamedArray instance of shape (# vars, nx, ny, nz).

    Args:
        names (Iterable): Series of names, must be valid Python variable names,
                          corresponding to indices of a NamedArray instance.
        x (Tuple[dict, dict]):
            Boundary condition type in the x-direction, specified at either boundary
            for each variable. Possible formats:
                - ({"var1": bc_var1_left, ...}, {"var1": bc_var1_right, ...})
                - (bc_left, bc_right): Applies the same BC to all variables.
                - bc: Applies the same BC to all variables on both sides.
            see VALID_BC_TYPES for valid boundary condition types.
        y (Tuple[dict, dict]): Boundary condition type in the y-direction.
        z (Tuple[dict, dict]): Boundary condition type in the z-direction.
        x_value (Tuple[dict, dict]):
            Boundary condition value in the x-direction, specified at either boundary
            for each variable. Possible formats:
                - ({"var1": bc_val1_left, ...}, {"var1": bc_val1_right, ...})
                - (bc_left, bc_right): Applies the same BC value to all variables.
                - bc: Applies the same BC value to all variables on both sides.
        y_value (Tuple[dict, dict]): Boundary condition value in the y-direction.
        z_value (Tuple[dict, dict]): Boundary condition value in the z-direction.
        x_domain (Tuple[float, float]): Domain boundaries in the x-direction (x1, x2).
        y_domain (Tuple[float, float]): Domain boundaries in the y-direction (y1, y2).
        z_domain (Tuple[float, float]): Domain boundaries in the z-direction (z1, z2).
        h (Tuple[float, float, float]): Mesh spacings (hx, hy, hz).
    """

    names: Iterable = ()
    x: Tuple[dict, dict] = "periodic"
    y: Tuple[dict, dict] = "periodic"
    z: Tuple[dict, dict] = "periodic"
    x_value: Tuple[dict, dict] = None
    y_value: Tuple[dict, dict] = None
    z_value: Tuple[dict, dict] = None
    x_domain: Tuple[float, float] = (None, None)
    y_domain: Tuple[float, float] = (None, None)
    z_domain: Tuple[float, float] = (None, None)
    h: Tuple[float, float, float] = (None, None, None)

    def __post_init__(self):
        if len(self.names) == 0:
            return

        # configure bc types
        for dim in {"x", "y", "z"}:
            bc_type = getattr(self, dim)
            if bc_type is None or isinstance(bc_type, str) or isinstance(bc_type, dict):
                bc_type = (bc_type, bc_type)
            bc_type = list(bc_type)  # make mutable
            if len(bc_type) != 2:
                raise ValueError(f"Specify one or two boundary conditions for {dim}.")
            if (bc_type[0] == "periodic") + (bc_type[1] == "periodic") == 1:
                raise ValueError(
                    "Periodic boundary conditions must be specified for both boundaries."
                )
            for i in [0, 1]:
                if bc_type[i] == "reflective":
                    if set(self.names) != set(conservative_names):
                        raise NotImplementedError(
                            "Reflective boundary conditions are only implemented for conservative variables."
                        )
                    bc_type[i] = {
                        "rho": "symmetric",
                        "E": "symmetric",
                        "mx": "-symmetric" if dim == "x" else "symmetric",
                        "my": "-symmetric" if dim == "y" else "symmetric",
                        "mz": "-symmetric" if dim == "z" else "symmetric",
                    }
                elif bc_type[i] is None or isinstance(bc_type[i], str):
                    bc_type[i] = {var: bc_type[i] for var in self.names}
                for var in self.names:
                    if bc_type[i][var] not in VALID_BC_TYPES:
                        raise ValueError(
                            f"Invalid boundary condition '{bc_type[i][var]}' for variable '{var}'"
                        )
            setattr(self, dim, tuple(bc_type))  # return immutable

        # configure bc values
        for dim in {"x", "y", "z"}:
            bc_type = getattr(self, dim)
            bc_value = getattr(self, f"{dim}_value")
            if (
                bc_value is None
                or isinstance(bc_value, int)
                or isinstance(bc_value, float)
                or callable(bc_value)
                or isinstance(bc_value, dict)
            ):
                bc_value = (bc_value, bc_value)
            bc_value = list(bc_value)  # make mutable
            if len(bc_value) != 2:
                raise ValueError(
                    f"Specify one or two boundary condition values for {dim}."
                )
            for i in [0, 1]:
                if (
                    bc_value[i] is None
                    or isinstance(bc_value[i], int)
                    or isinstance(bc_value[i], float)
                    or callable(bc_value[i])
                ):
                    bc_value[i] = {var: bc_value[i] for var in self.names}
                for var in self.names:
                    # bc_value should be None if bc_type is not "dirichlet"
                    if bc_type[i][var] != "dirichlet":
                        bc_value[i][var] = None
                    elif bc_value[i][var] is None:
                        raise ValueError(
                            f"Boundary condition value must be specified for dirichlet boundary condition for variable '{var}'"
                        )
            setattr(self, f"{dim}_value", tuple(bc_value))  # return immutable

    def apply(
        self,
        u: NamedNumpyArray,
        gw: Iterable[int],
        t: float = None,
    ) -> np.ndarray:
        """
        args:
            u (NamedArray) : array of shape (# vars, nx, ny, nz)
            gw (Iterable[int]) : ghost zone width in each direction (gwx, gwy, gwz)
            t (float) : for time-dependent boundary conditions
        returns:
            out (array_like) : w with bcs applied, shape (5, nx + 2 * gwx, ...)
        """
        if np.any(np.isnan(u)):
            raise ValueError(
                "Boundary conditions cannot be applied to arrays with NaNs."
            )

        # apply temporary boundaries of nan
        out = np.pad(
            u,
            pad_width=[(0, 0), (gw[0], gw[0]), (gw[1], gw[1]), (gw[2], gw[2])],
            mode="constant",
            constant_values=np.nan,
        )
        out = u.__class__(input_array=out, names=u.variable_names)

        # define domain
        all_bc_types = (
            set(self.x[0].values())
            | set(self.x[1].values())
            | set(self.y[0].values())
            | set(self.y[1].values())
            | set(self.z[0].values())
            | set(self.z[1].values())
        )
        if "dirichlet" in all_bc_types:
            nx, ny, nz = out.shape[1:]
            X, Y, Z = uniform_fv_mesh(
                nx=nx,
                ny=ny,
                nz=nz,
                x=(
                    self.x_domain[0] - self.h[0] * gw[0],
                    self.x_domain[1] + self.h[0] * gw[0],
                ),
                y=(
                    self.y_domain[0] - self.h[1] * gw[1],
                    self.y_domain[1] + self.h[1] * gw[1],
                ),
                z=(
                    self.z_domain[0] - self.h[2] * gw[2],
                    self.z_domain[1] + self.h[2] * gw[2],
                ),
            )
            if u.xp == "cupy" and CUPY_AVAILABLE:
                X = cp.asarray(X)
                Y = cp.asarray(Y)
                Z = cp.asarray(Z)

        # define views for each boundary region
        ALREADY_APPLIED_PERIODIC_BOUNDARY = {}
        for var, (j, dim), (i, pos) in product(
            u.variable_names, enumerate(["x", "y", "z"]), enumerate(["l", "r"])
        ):
            bc = getattr(self, dim)[i][var]
            bc_value = getattr(self, f"{dim}_value")[i][var]
            num_ghost = gw[j]

            if num_ghost == 0:
                continue

            match bc:
                case "dirichlet":
                    ubc = set_dirichlet_bc(
                        f=bc_value,
                        u=getattr(out, var),
                        num_ghost=num_ghost,
                        dim=dim,
                        pos=pos,
                        x=X,
                        y=Y,
                        z=Z,
                        t=t,
                    )
                case "free":
                    ubc = set_free_bc(
                        u=getattr(out, var), num_ghost=num_ghost, dim=dim, pos=pos
                    )
                case "periodic":
                    if ALREADY_APPLIED_PERIODIC_BOUNDARY.get(f"{var}{dim}", False):
                        continue
                    ubc = set_periodic_bc(
                        u=getattr(out, var), num_ghost=num_ghost, dim=dim
                    )
                    ALREADY_APPLIED_PERIODIC_BOUNDARY[f"{var}{dim}"] = True
                case "special-case-double-mach-reflection-y=0":
                    ubc_free = set_free_bc(
                        u=getattr(out, var), num_ghost=num_ghost, dim=dim, pos=pos
                    )
                    ubc_reflective = set_symmetric_bc(
                        u=getattr(out, var),
                        num_ghost=num_ghost,
                        dim=dim,
                        pos=pos,
                        negative=var == "my",
                    )
                    ubc = np.where(X < 1 / 6, ubc_free, ubc_reflective)
                case "symmetric":
                    ubc = set_symmetric_bc(
                        u=getattr(out, var), num_ghost=num_ghost, dim=dim, pos=pos
                    )
                case "-symmetric":
                    ubc = set_symmetric_bc(
                        u=getattr(out, var),
                        num_ghost=num_ghost,
                        dim=dim,
                        pos=pos,
                        negative=True,
                    )
                case None:
                    continue
                case _:
                    raise TypeError(f"Invalid boundary condition '{bc}'")

            setattr(out, var, ubc)

        if np.any(np.isnan(out)):
            raise ValueError("Boundary conditions not applied correctly.")

        return out

    def _json_dict(self):
        """
        returns a json dictionary of the boundary conditions
        """
        if hasattr(self, "json_dict"):
            return self.json_dict
        x_value = self.x_value
        x_value = (
            {
                k: v.__name__ + "_func" if callable(v) else v
                for k, v in x_value[0].items()
            },
            {
                k: v.__name__ + "_func" if callable(v) else v
                for k, v in x_value[1].items()
            },
        )
        y_value = self.y_value
        y_value = (
            {
                k: v.__name__ + "_func" if callable(v) else v
                for k, v in y_value[0].items()
            },
            {
                k: v.__name__ + "_func" if callable(v) else v
                for k, v in y_value[1].items()
            },
        )
        z_value = self.z_value
        z_value = (
            {
                k: v.__name__ + "_func" if callable(v) else v
                for k, v in z_value[0].items()
            },
            {
                k: v.__name__ + "_func" if callable(v) else v
                for k, v in z_value[1].items()
            },
        )
        json_dict = {
            "names": self.names,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "x_value": x_value,
            "y_value": y_value,
            "z_value": z_value,
            "x_domain": self.x_domain,
            "y_domain": self.y_domain,
            "z_domain": self.z_domain,
            "h": self.h,
        }
        self._json_dict = json_dict
        return json_dict
