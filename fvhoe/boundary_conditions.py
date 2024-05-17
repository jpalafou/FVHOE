from dataclasses import dataclass
from functools import partial
from fvhoe.fv import get_window
import numpy as np
from typing import Iterable, Tuple, Union


def parse_bc(s: str, splitchar: str = "-") -> Union[str, Tuple[str, str]]:
    """
    args:
        s (str): boundary condition-specifying string. may contain up to 1 splitchar
        splitchar (str) : for separating left/right boundary conditions
    returns:
        out (Tuple[str, str]) : (left boundary condition type, right boundary condition type) or "periodic"
    """
    if s == "periodic":
        return s
    s_split = s.split(splitchar)
    if len(s_split) == 1:
        out = (s, s)
    elif len(s_split) == 2:
        out = tuple(s_split)
    else:
        raise TypeError("Specify 1 or 2 boundary condition types")
    for bct in s_split:
        if bct not in ("dirichlet", "neumann"):
            raise TypeError(f"Invalid boundary condition type {s}")
    return out


def set_finite_difference_bc(
    u: np.ndarray,
    ng: int,
    p: int,
    h: float,
    axis: int,
    pos: str,
    slope: float = 0.0,
) -> None:
    """
    set boundaries of u to satistfy a Neumann boundary with a degree p polynomial
    args:
        u (array_like) : array of values
        ng (int) : number of 'ghost zones' on  pos end of domain
        p (int) : interpolation polynomial degree
        h (float) : grid spacing along axis
        axis (int) : along which to apply bcs
        pos (str) : either end of domain
            "l" left boundary along axis
            "r" right boundary along axis
        slope (float) : target slopes at the pos end along axis
    returns:
        None : revises u
    """

    gw = partial(get_window, ndim=u.ndim, axis=axis)

    if pos == "l":
        u = u[gw(step=-1)]
        set_finite_difference_bc(
            u=u,
            ng=ng,
            p=p,
            h=h,
            axis=axis,
            pos="r",
            slope=-slope,
        )
        u = u[gw(step=-1)]
        return
    if p == 0:
        return
    elif p in (1, 2):
        # u_i+1 = 2 slope h + u_i-1
        G = 2 * slope * h
        for n in range(ng)[::-1]:
            u[gw(cut=(-n - 1, n))] = G + u[gw(cut=(-n - 3, n + 2))]
    elif p in (3, 4):
        # u_i+2 = -12 slope h + 8 u_i+1 - 8u_i-1 + u_i-2
        G = -12 * slope * h
        for n in range(ng)[::-1]:
            u[gw(cut=(-n - 1, n))] = (
                G
                + 8 * u[gw(cut=(-n - 2, n + 1))]
                - 8 * u[gw(cut=(-n - 4, n + 3))]
                + u[gw(cut=(-n - 5, n + 4))]
            )
    else:
        raise NotImplementedError(f"{p=}")


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

    gw = partial(get_window, ndim=u.ndim, axis=axis)

    if p == 0:
        out = u.copy()
    elif p in (1, 2):
        out = (1 / (2 * h)) * (1 * u[gw(cut=(2, 0))] + -1 * u[gw(cut=(0, 2))])
    elif p in (3, 4):
        out = (1 / (12 * h)) * (
            -1 * u[gw(cut=(4, 0))]
            + 8 * u[gw(cut=(3, 1))]
            + -8 * u[gw(cut=(1, 3))]
            + 1 * u[gw(cut=(0, 4))]
        )
    else:
        raise NotImplementedError(f"{p=}")
    return out


@dataclass
class BoundaryCondition:
    """
    boundary condition class for a NamedArray instance of shape (# vars, nx, ny, nz)
    args:
        NamedArray (callable): NamedArray init
        x (str) : boundary condition type in x-direciton
            "periodic" : periodic boundary condition
            "dirichlet" : dirichlet boundary condition at both ends
            "neumann" : neumann boundary condition at both ends
            "{bc_l}-{bc_r}" : one boundary type on the left, another on the right
        y (str) : boundary condition type in y-direciton
        z (str) : boundary condition type in z-direciton
        names (Iterable) : series of names, must be valid python variable names
    """

    NamedArray: callable
    x: str = "periodic"
    y: str = "periodic"
    z: str = "periodic"
    names: Iterable = ()

    def __post_init__(self):
        self.x = parse_bc(self.x)
        self.y = parse_bc(self.y)
        self.z = parse_bc(self.z)

    def apply(self, w: np.ndarray, gw: Iterable[int]) -> np.ndarray:
        """
        args:
            w (array_like) : array of primitive variables of shape (5, nx, ny, nz)
            gw (Iterable[int]) : ghost zone width in each direction (gwx, gwy, gwz)
        returns:
            out (array_like) : w with bcs applied, shape (5, nx + 2 * gwx, ...)
        """
        out = w
        for axis, dim in zip(range(1, 4), ["x", "y", "z"]):
            # specify pad width
            pad_width = [(0, 0), (0, 0), (0, 0), (0, 0)]
            pad_width[axis] = (gw[axis - 1], gw[axis - 1])

            if getattr(self, dim) == "periodic":
                out = np.pad(out, pad_width=pad_width, mode="wrap")
                continue

        out = self.NamedArray(input_array=out, names=w.variable_names)
        return out
