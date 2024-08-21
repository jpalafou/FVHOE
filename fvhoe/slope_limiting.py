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
from typing import Tuple

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
    NAD_tolerance: float = 1e-5,
    NAD_mode: str = "global",
    PAD: dict = None,
    SED: bool = True,
    SED_tolerance: float = 1e-10,
    xp: str = "numpy",
) -> np.ndarray:
    """
    args:
        u (NamedArray) : array of values with shape (# variables, nx, ny, nz). if u is a NamedArray, the output will still be a numpy-like array
        u_candidate (NameArray) : array of candidate values with shape (# variables, nx, ny, nz)
        dims (str) : contains "x", "y", and/or "z"
        NAD_tolerance (float) : tolerance for NAD
        NAD_mode (str) : "global" or "local"
            "global" : NAD is applied based on the global range of each variable
            "local" : NAD is applied based on the local range of each variable
        global_NAD: (bool) :
        PAD (dict) : dictionary of PAD parameters with keys given by the variables in u
        SED (bool) : remove NAD trouble flags where a smooth extremum is detected
        SED_tolerance (float) : tolerance for avoiding dividing by 0 in smooth extrema detection
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

    # take minimum of neighbors
    footprint = np.ones(footprint_shape)
    if xp == "numpy" or not CUPY_AVAILABLE:
        M = maximum_filter(u, footprint=footprint, mode="constant", cval=0)[
            interior_slice
        ]
        m = minimum_filter(u, footprint=footprint, mode="constant", cval=0)[
            interior_slice
        ]
    elif xp == "cupy":
        M = cp_maximum_filter(u, footprint=footprint, mode="constant", cval=0)[
            interior_slice
        ]
        m = cp_minimum_filter(u, footprint=footprint, mode="constant", cval=0)[
            interior_slice
        ]
    else:
        raise ValueError(f"Unknown xp: {xp}")

    # get relevant view of u_candidate
    u_candidate_inner = u_candidate[interior_slice]

    # get variable ranges
    if NAD_mode == "global":
        # global NAD for each variable
        u_range = np.max(u, axis=(1, 2, 3), keepdims=True) - np.min(
            u, axis=(1, 2, 3), keepdims=True
        )
        u_range = u.__class__(u_range, u.variable_names)
    elif NAD_mode == "local":
        # local NAD for each variable
        u_range = M - m
    else:
        raise ValueError(f"Unknown NAD_mode: {NAD_mode}")

    # compute NAD trouble per variable
    tolerance_per_var = NAD_tolerance * u_range
    local_undershoot = u_candidate_inner - m
    local_overshoot = M - u_candidate_inner
    NAD_indicator_per_var = np.minimum(local_undershoot, local_overshoot)
    NAD_trouble_per_var = NAD_indicator_per_var < -tolerance_per_var

    # smooth extrema detection per variable
    if SED:
        if len(dims) == 1:
            alpha_per_var = compute_1d_smooth_extrema_detector(
                u_candidate, dim=dims, eps=SED_tolerance
            )
        elif len(dims) == 2:
            alpha_per_var = compute_2d_smooth_extrema_detector(
                u_candidate, dims=dims, eps=SED_tolerance
            )
        elif len(dims) == 3:
            alpha_per_var = compute_3d_smooth_extrema_detector(
                u_candidate, eps=SED_tolerance
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
    PAD_trouble = np.zeros_like(NAD_trouble, dtype=bool)
    PAD_violation_magnitude = np.zeros_like(NAD_trouble, dtype=float)
    if PAD is not None:
        for var in u_candidate_inner.variable_names:
            lower_PAD_difference = getattr(u_candidate_inner, var) - PAD[var][0]
            upper_PAD_difference = PAD[var][1] - getattr(u_candidate_inner, var)
            PAD_difference = np.minimum(lower_PAD_difference, upper_PAD_difference)
            PAD_trouble[...] = np.where(PAD_difference < 0, 1, PAD_trouble)
            PAD_violation_magnitude[...] = np.maximum(
                np.where(PAD_difference < 0, -PAD_difference, 0),
                PAD_violation_magnitude,
            )

    trouble = np.where(PAD_trouble, 1, NAD_trouble)

    return trouble, NAD_violation_magnitude


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
    troubled_x_interfaces = np.zeros((nx + 1, ny, nz))
    troubled_y_interfaces = np.zeros((nx, ny + 1, nz))
    troubled_z_interfaces = np.zeros((nx, ny, nz + 1))
    if convex:
        xdim, ydim, zdim = "x" in dims, "y" in dims, "z" in dims
        nx_alloc = trouble.shape[0] + 4 if xdim else 1
        ny_alloc = trouble.shape[1] + 4 if ydim else 1
        nz_alloc = trouble.shape[2] + 4 if zdim else 1
        alloc_trouble = np.zeros((nx_alloc, ny_alloc, nz_alloc))
        slices = [
            slice(2, -2) if xdim else slice(None),
            slice(2, -2) if ydim else slice(None),
            slice(2, -2) if zdim else slice(None),
        ]
        alloc_trouble[tuple(slices)] = trouble
        # apply periodic boundary conditions
        if periodic_x and xdim:
            alloc_trouble[:2, :, :] = trouble[-4:-2, :, :]
            alloc_trouble[-2:, :, :] = trouble[2:4, :, :]
        if periodic_y and ydim:
            alloc_trouble[:, :2, :] = trouble[:, -4:-2, :]
            alloc_trouble[:, -2:, :] = trouble[:, 2:4, :]
        if periodic_z and zdim:
            alloc_trouble[:, :, :2] = trouble[:, :, -4:-2]
            alloc_trouble[:, :, -2:] = trouble[:, :, 2:4]
    # convert to cupy if necessary
    if xp == "cupy" and CUPY_AVAILABLE:
        troubled_x_interfaces = cp.asarray(troubled_x_interfaces)
        troubled_y_interfaces = cp.asarray(troubled_y_interfaces)
        troubled_z_interfaces = cp.asarray(troubled_z_interfaces)
        if convex:
            alloc_trouble = cp.asarray(alloc_trouble)
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
            convex_troubled_x_interfaces, convex_troubled_y_interfaces = (
                convex_2d_broadcast_to_troubled_interfaces(alloc_trouble_2d)
            )
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
