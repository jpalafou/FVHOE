from functools import partial
from fvhoe.fv import get_view
from fvhoe.named_array import NamedNumpyArray
from fvhoe.smooth_extrema_detection import (
    compute_1d_smooth_extrema_detector,
    compute_2d_smooth_extrema_detector,
    compute_3d_smooth_extrema_detector,
)
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from typing import Dict, Tuple

try:
    import cupy as cp
    from cupyx.scipy.ndimage import maximum_filter as cp_maximum_filter
    from cupyx.scipy.ndimage import minimum_filter as cp_minimum_filter

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


def MUSCL_interpolations(
    u: np.ndarray, axis: int, limiter: str = "minmod"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns the slope-limited, p=1 polynomial reconstructions of a given fv array along a specified dimension
    args:
        u (np.ndarray) : array of values with shape (# variables, nx, ny, nz)
        axis (int): axis along which to interpolate
        limiter (str): slope limiter "minmod", "moncen", None (2nd order central diffserence)
    returns:
        left_face, right_face (Tuple[np.ndarray, np.ndarray]) : MUSCL reconstructions
    """
    gv = partial(get_view, ndim=u.ndim, axis=axis)
    du_left = u[gv(cut=(1, 1))] - u[gv(cut=(0, 2))]
    du_right = u[gv(cut=(2, 0))] - u[gv(cut=(1, 1))]

    match limiter:
        case "minmod":
            limited_slopes = minmod(du_left, du_right)
        case "moncen":
            limited_slopes = moncen(du_left, du_right)
        case None:
            limited_slopes = 0.5 * (du_left + du_right)
        case _:
            raise ValueError(f"Unknown slope limiter: {limiter}")

    left_face = u[gv(cut=(1, 1))] - 0.5 * limited_slopes
    right_face = u[gv(cut=(1, 1))] + 0.5 * limited_slopes

    return left_face, right_face


def minmod(du_left: np.ndarray, du_right: np.ndarray) -> np.ndarray:
    """
    args:
        du_left:    left difference
        du_right:   right difference
    returns:
        minmod limited difference
    """
    ratio = du_right / np.where(
        du_left > 0,
        np.where(du_left > 1e-16, du_left, 1e-16),
        np.where(du_left < -1e-16, du_left, -1e-16),
    )
    ratio = np.where(ratio < 1, ratio, 1)
    return np.where(ratio > 0, ratio, 0) * du_left


def moncen(du_left: np.ndarray, du_right: np.ndarray) -> np.ndarray:
    """
    args:
        du_left:    left difference
        du_right:   right difference
    returns:
        moncen limited difference
    """
    du_central = 0.5 * (du_left + du_right)
    slope = np.minimum(np.abs(2 * du_left), np.abs(2 * du_right))
    slope = np.sign(du_central) * np.minimum(slope, np.abs(du_central))
    return np.where(du_left * du_right >= 0, slope, 0)


def detect_troubled_cells(
    u: NamedNumpyArray,
    u_candidate: NamedNumpyArray,
    dims: str = "xyz",
    NAD_eps: float = 1e-5,
    mode: str = "global",
    range_type: str = "relative",
    NAD_vars: list = None,
    PAD_bounds: Dict[str, Tuple[float, float]] = None,
    SED: bool = True,
    SED_eps: float = 1e-10,
    xp: str = "numpy",
) -> np.ndarray:
    """
    args:
        u (NamedArray) : array of values with shape (# variables, nx, ny, nz). if u is a NamedArray, the output will still be a numpy-like array
        u_candidate (NameArray) : array of candidate values with shape (# variables, nx, ny, nz)
        dims (str) : contains "x", "y", and/or "z"
        NAD_eps (float) : tolerance for NAD
        mode (str) : "global" or "local"
            "global" : NAD is applied based on the global range of each variable
            "local" : NAD is applied based on the local range of each variable
        range_type (str) : "relative" or "absolute"
            "relative" : NAD is applied based on the relative range of each variable
                upper_bound = max + (max - min) * eps
                lower_bound = min - (max - min) * eps
            "absolute" : NAD is applied based on the absolute range of each variable
                upper_bound = (1 + eps) * max
                lower_bound = (1 - eps) * min
        NAD_vars (list) : list of variables to apply NAD. if None, all variables are considered
        PAD_bounds (dict) : dictionary of PAD parameters. keys are variable names, values are tuples of lower and upper bounds, e.g. {"u": (0, 1)}. PAD is not applied if None
        SED (bool) : remove NAD trouble flags where a smooth extremum is detected
        SED_eps (float) : tolerance for avoiding dividing by 0 in smooth extrema detection
        xp (str) : 'numpy' or 'cupy'
    returns:
        trouble (array_like) : array of troubled cells indicated by 1, shape (nx - 2, ny - 2, nz - 2)
        NAD_violation_magnitude (array_like) : array of NAD violation magnitude, shape (nx - 2, ny - 2, nz - 2)
    """
    # define views in specified dimensions
    footprint_shape = [1, 1, 1, 1]
    interior_slice = [slice(None), slice(None), slice(None), slice(None)]
    if "x" in dims:
        footprint_shape[1] = 3
        interior_slice[1] = slice(3, -3)
    if "y" in dims:
        footprint_shape[2] = 3
        interior_slice[2] = slice(3, -3)
    if "z" in dims:
        footprint_shape[3] = 3
        interior_slice[3] = slice(3, -3)
    footprint_shape = tuple(footprint_shape)
    interior_slice = tuple(interior_slice)

    # filter views
    if NAD_vars is None:
        u_copy = u.copy()
        u_candidate_copy = u_candidate.copy()
    else:
        u_copy = u.filter(NAD_vars)
        u_candidate_copy = u_candidate.filter(NAD_vars)
    u_candidate_inner = u_candidate_copy[interior_slice]

    # take maximum and minimum of neighbors
    footprint = np.ones(footprint_shape)
    if xp == "numpy" or not CUPY_AVAILABLE:
        M = maximum_filter(u_copy, footprint=footprint, mode="constant", cval=0)[
            interior_slice
        ]
        m = minimum_filter(u_copy, footprint=footprint, mode="constant", cval=0)[
            interior_slice
        ]
    elif xp == "cupy":
        M = cp_maximum_filter(u_copy, footprint=footprint, mode="constant", cval=0)[
            interior_slice
        ]
        m = cp_minimum_filter(u_copy, footprint=footprint, mode="constant", cval=0)[
            interior_slice
        ]
    else:
        raise ValueError(f"Unknown xp: {xp}")

    # compute NAD indicator and trouble flags
    if range_type == "relative":
        if mode == "global":
            # relative global NAD
            u_range = np.max(u_copy, axis=(1, 2, 3), keepdims=True) - np.min(
                u_copy, axis=(1, 2, 3), keepdims=True
            )
            u_range = u_copy.__class__(u_range, u_copy.variable_names)
        elif mode == "local":
            # relative local NAD
            u_range = M - m
        else:
            raise ValueError(f"Unknown mode: {mode}")
        NAD_indicator_per_var = u_candidate_inner - m  # local undershoot
        NAD_indicator_per_var[...] = np.minimum(
            NAD_indicator_per_var, M - u_candidate_inner
        )  # local overshoot
        NAD_trouble_per_var = NAD_indicator_per_var < -NAD_eps * u_range
    elif range_type == "absolute":
        if mode == "global":
            # absolute global NAD
            lower_bound = (1 - NAD_eps) * np.min(u_copy, axis=(1, 2, 3), keepdims=True)
            upper_bound = (1 + NAD_eps) * np.max(u_copy, axis=(1, 2, 3), keepdims=True)
        elif mode == "local":
            # absolute local NAD
            lower_bound = (1 - NAD_eps) * m
            upper_bound = (1 + NAD_eps) * M
        else:
            raise ValueError(f"Unknown mode: {mode}")
        NAD_indicator_per_var = np.minimum(
            u_candidate_inner - lower_bound, upper_bound - u_candidate_inner
        )
        NAD_trouble_per_var = NAD_indicator_per_var < 0.0
    else:
        raise ValueError(f"Unknown range_type: {range_type}")

    # smooth extrema detection per variable
    if SED:
        if len(dims) == 1:
            alpha_per_var = compute_1d_smooth_extrema_detector(
                u_candidate_copy, dim=dims, eps=SED_eps
            )
        elif len(dims) == 2:
            alpha_per_var = compute_2d_smooth_extrema_detector(
                u_candidate_copy, dims=dims, eps=SED_eps
            )
        elif len(dims) == 3:
            alpha_per_var = compute_3d_smooth_extrema_detector(
                u_candidate_copy, eps=SED_eps
            )
        else:
            raise ValueError(f"Invalid dims specified: {dims}")
        NAD_trouble_per_var[...] = np.where(alpha_per_var < 1, NAD_trouble_per_var, 0)

    # NAD across all variables
    NAD_trouble = np.any(NAD_trouble_per_var, axis=0)

    # store NAD violation magnitude
    NAD_indicator = np.min(NAD_indicator_per_var, axis=0)
    NAD_violation_magnitude = np.where(NAD_trouble, -NAD_indicator, 0)

    # PAD
    PAD_indicator = np.zeros_like(NAD_violation_magnitude)
    if PAD_bounds is not None:
        for var, (lowr, uppr) in PAD_bounds.items():
            if var not in u_candidate_inner.variable_names:
                raise ValueError(f"Variable {var} not found in u")
            PAD_indicator[...] = (
                getattr(u_candidate_inner, var) - lowr
            )  # lower PAD difference
            PAD_indicator[...] = np.minimum(
                PAD_indicator, uppr - getattr(u_candidate_inner, var)
            )  # upper PAD difference
    PAD_trouble = np.where(PAD_indicator < 0.0, 1, 0)
    PAD_violation_magnitude = np.where(PAD_trouble, -PAD_indicator, 0)

    # combine NAD and PAD
    trouble = np.where(PAD_trouble, 1, NAD_trouble)

    return trouble, NAD_violation_magnitude, PAD_violation_magnitude


def broadcast_to_troubled_interfaces(
    trouble: np.ndarray,
    dims: str = None,
    convex: bool = False,
    periodic_x: bool = False,
    periodic_y: bool = False,
    periodic_z: bool = False,
    xp: str = "numpy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    args:
        trouble (array_like) : array of troubled cells indicated by 1, shape (nx, ny, nz)
        dims (str) : dimensions to broadcast troubled cells
        convex (bool) : convex broadcast
        periodic_x (bool) : periodic boundary conditions in x
        periodic_y (bool) : periodic boundary conditions in y
        periodic_z (bool) : periodic boundary conditions in z
        xp (str) : 'numpy' or 'cupy'
    returns:
        troubled_x_interfaces (array_like) : array of troubled x interfaces indicated by 1, shape (1, nx + 1, ny, nz)
        troubled_y_interfaces (array_like) : array of troubled y interfaces indicated by 1, shape (1, nx, ny + 1, nz)
        troubled_z_interfaces (array_like) : array of troubled z interfaces indicated by 1, shape (1, nx, ny, nz + 1)
    """
    # allocate troubled interface arrays
    nx, ny, nz = trouble.shape
    using_cupy = xp == "cupy" and CUPY_AVAILABLE
    if using_cupy:
        troubled_x_interfaces = cp.zeros((nx + 1, ny, nz))
        troubled_y_interfaces = cp.zeros((nx, ny + 1, nz))
        troubled_z_interfaces = cp.zeros((nx, ny, nz + 1))
    else:
        troubled_x_interfaces = np.zeros((nx + 1, ny, nz))
        troubled_y_interfaces = np.zeros((nx, ny + 1, nz))
        troubled_z_interfaces = np.zeros((nx, ny, nz + 1))
    if convex:
        xdim, ydim, zdim = "x" in dims, "y" in dims, "z" in dims
        nx_alloc = trouble.shape[0] + 4 if xdim else 1
        ny_alloc = trouble.shape[1] + 4 if ydim else 1
        nz_alloc = trouble.shape[2] + 4 if zdim else 1
        if using_cupy:
            alloc_trouble = cp.zeros((nx_alloc, ny_alloc, nz_alloc))
        else:
            alloc_trouble = np.zeros((nx_alloc, ny_alloc, nz_alloc))
        slices = [
            slice(2, -2) if xdim else slice(None),
            slice(2, -2) if ydim else slice(None),
            slice(2, -2) if zdim else slice(None),
        ]
        alloc_trouble[tuple(slices)] = trouble
        # apply periodic boundary conditions
        if periodic_x and xdim:
            alloc_trouble[:2, 2:-2, 2:-2] = trouble[-2:, :, :]
            alloc_trouble[-2:, 2:-2, 2:-2] = trouble[:2, :, :]
        if periodic_y and ydim:
            alloc_trouble[2:-2, :2, 2:-2] = trouble[:, -2:, :]
            alloc_trouble[2:-2, -2:, 2:-2] = trouble[:, :2, :]
        if periodic_z and zdim:
            alloc_trouble[2:-2, 2:-2, :2] = trouble[2:-2, 2:-2, -2:]
            alloc_trouble[2:-2, 2:-2, -2:] = trouble[2:-2, 2:-2, :2]
    # broadcast troubled cells to interfaces
    if convex:
        # convex broadcast
        ndim = sum([xdim, ydim, zdim])
        slices = [0, 0, 0]
        if ndim == 1:
            slices[{"x": 0, "y": 1, "z": 2}[dims]] = slice(None)
            alloc_trouble_1d = alloc_trouble[tuple(slices)]
            convex_troubled_x_interfaces = convex_1d_broadcast_to_troubled_interfaces(
                alloc_trouble_1d
            )
            if dims == "x":
                troubled_x_interfaces[...] = convex_troubled_x_interfaces.reshape(
                    -1, 1, 1
                )
            elif dims == "y":
                troubled_y_interfaces[...] = convex_troubled_x_interfaces.reshape(
                    1, -1, 1
                )
            elif dims == "z":
                troubled_z_interfaces[...] = convex_troubled_x_interfaces.reshape(
                    1, 1, -1
                )
            else:
                raise ValueError(f"Invalid dims specified: {dims}")
        elif ndim == 2:
            slices[{"x": 0, "y": 1, "z": 2}[dims[0]]] = slice(None)
            slices[{"x": 0, "y": 1, "z": 2}[dims[1]]] = slice(None)
            alloc_trouble_2d = alloc_trouble[tuple(slices)]
            (
                convex_troubled_x_interfaces,
                convex_troubled_y_interfaces,
            ) = convex_2d_broadcast_to_troubled_interfaces(alloc_trouble_2d)
            if dims in ["xy", "yx"]:
                troubled_x_interfaces[...] = convex_troubled_x_interfaces.reshape(
                    nx + 1, ny, 1
                )
                troubled_y_interfaces[...] = convex_troubled_y_interfaces.reshape(
                    nx, ny + 1, 1
                )
            elif dims in ["yz", "zy"]:
                troubled_y_interfaces[...] = convex_troubled_x_interfaces.reshape(
                    1, ny + 1, nz
                )
                troubled_z_interfaces[...] = convex_troubled_y_interfaces.reshape(
                    1, ny, nz + 1
                )
            elif dims in ["xz", "zx"]:
                troubled_x_interfaces[...] = convex_troubled_x_interfaces.reshape(
                    nx + 1, 1, nz
                )
                troubled_z_interfaces[...] = convex_troubled_y_interfaces.reshape(
                    nx, 1, nz + 1
                )
            else:
                raise ValueError(f"Invalid dims specified: {dims}")
        elif ndim == 3:
            raise NotImplementedError("3D convex broadcast not implemented")
    else:
        # simple broadcast
        if "x" in dims:
            troubled_x_interfaces[1:, :, :] = trouble
            troubled_x_interfaces[:-1, :, :] = np.maximum(
                troubled_x_interfaces[:-1, :, :], trouble
            )
        if "y" in dims:
            troubled_y_interfaces[:, 1:, :] = trouble
            troubled_y_interfaces[:, :-1, :] = np.maximum(
                troubled_y_interfaces[:, :-1, :], trouble
            )
        if "z" in dims:
            troubled_z_interfaces[:, :, 1:] = trouble
            troubled_z_interfaces[:, :, :-1] = np.maximum(
                troubled_z_interfaces[:, :, :-1], trouble
            )
    troubled_x_interfaces = troubled_x_interfaces[np.newaxis, ...]
    troubled_y_interfaces = troubled_y_interfaces[np.newaxis, ...]
    troubled_z_interfaces = troubled_z_interfaces[np.newaxis, ...]
    return troubled_x_interfaces, troubled_y_interfaces, troubled_z_interfaces


def convex_1d_broadcast_to_troubled_interfaces(trouble: np.ndarray):
    """
    args:
        trouble (array_like) : array of troubled cells indicated by 1, shape (nx + 4,)
    returns:
        troubled_interfaces (array_like) : array of troubled interfaces indicated by 1, shape (nx + 1,)
    """
    theta = trouble.copy()

    # First neighbors
    theta[:-1] = np.maximum(0.75 * trouble[1:], theta[:-1])
    theta[1:] = np.maximum(0.75 * trouble[:-1], theta[1:])

    # Second neighbors
    theta[:-1] = np.maximum(0.25 * (theta[1:] > 0), theta[:-1])
    theta[1:] = np.maximum(0.25 * (theta[:-1] > 0), theta[1:])

    # flag affected faces with theta
    troubled_interfaces = np.maximum(theta[1:-2], theta[2:-1])

    return troubled_interfaces


def convex_2d_broadcast_to_troubled_interfaces(trouble: np.ndarray):
    """
    args:
        trouble (array_like) : array of troubled cells indicated by 1, shape (nx + 4, ny + 4)
    returns:
        troubled_x_interfaces (array_like) : array of troubled x interfaces indicated by 1, shape (nx + 1, ny)
        troubled_y_interfaces (array_like) : array of troubled y interfaces indicated by 1, shape (nx, ny + 1)
    """
    theta = trouble.copy()

    # First neighbors
    theta[:, :-1] = np.maximum(0.75 * trouble[:, 1:], theta[:, :-1])
    theta[:, 1:] = np.maximum(0.75 * trouble[:, :-1], theta[:, 1:])
    theta[:-1, :] = np.maximum(0.75 * trouble[1:, :], theta[:-1, :])
    theta[1:, :] = np.maximum(0.75 * trouble[:-1, :], theta[1:, :])

    # Second neighbors
    theta[:-1, :-1] = np.maximum(0.5 * trouble[1:, 1:], theta[:-1, :-1])
    theta[:-1, 1:] = np.maximum(0.5 * trouble[1:, :-1], theta[:-1, 1:])
    theta[1:, :-1] = np.maximum(0.5 * trouble[:-1, 1:], theta[1:, :-1])
    theta[1:, 1:] = np.maximum(0.5 * trouble[:-1, :-1], theta[1:, 1:])

    # Third neighbors
    theta[:, :-1] = np.maximum(0.25 * (theta[:, 1:] > 0), theta[:, :-1])
    theta[:, 1:] = np.maximum(0.25 * (theta[:, :-1] > 0), theta[:, 1:])
    theta[:-1, :] = np.maximum(0.25 * (theta[1:, :] > 0), theta[:-1, :])
    theta[1:, :] = np.maximum(0.25 * (theta[:-1, :] > 0), theta[1:, :])

    # flag affected faces with theta
    troubled_x_interfaces = np.maximum(theta[2:-1, 2:-2], theta[1:-2, 2:-2])
    troubled_y_interfaces = np.maximum(theta[2:-2, 2:-1], theta[2:-2, 1:-2])

    return troubled_x_interfaces, troubled_y_interfaces
