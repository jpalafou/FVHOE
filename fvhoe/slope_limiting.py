from functools import partial
from fvhoe.fv import get_view
from fvhoe.named_array import NamedNumpyArray
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
    eps: float = 1e-5,
    PAD: dict = None,
    xp: str = "numpy",
) -> np.ndarray:
    """
    args:
        u (NamedArray) : array of values with shape (# variables, nx, ny, nz). if u is a NamedArray, the output will still be a numpy-like array
        u_candidate (NameArray) : array of candidate values with shape (# variables, nx, ny, nz)
        eps (float) : tolerance for NAD
        PAD (dict) : dictionary of PAD parameters with keys given by the variables in u
        xp (str) : 'numpy' or 'cupy'
    returns:
        trouble (array_like) : array of troubled cells indicated by 1, shape (nx - 2, ny - 2, nz - 2)
    """
    footprint = np.ones((1, 3, 3, 3))
    if xp == "numpy" or not CUPY_AVAILABLE:
        M = maximum_filter(u, footprint=footprint, mode="constant", cval=0)[
            :, 1:-1, 1:-1, 1:-1
        ]
        m = minimum_filter(u, footprint=footprint, mode="constant", cval=0)[
            :, 1:-1, 1:-1, 1:-1
        ]
    elif xp == "cupy":
        M = cp_maximum_filter(u, footprint=footprint, mode="constant", cval=0)[
            :, 1:-1, 1:-1, 1:-1
        ]
        m = cp_minimum_filter(u, footprint=footprint, mode="constant", cval=0)[
            :, 1:-1, 1:-1, 1:-1
        ]
    else:
        raise ValueError(f"Unknown xp: {xp}")
    u_candidate_inner = u_candidate[:, 1:-1, 1:-1, 1:-1]

    # NAD
    u_range = np.max(u, axis=(1, 2, 3)) - np.min(u, axis=(1, 2, 3))
    tolerance = (eps * u_range)[:, np.newaxis, np.newaxis, np.newaxis]
    upper_differences = u_candidate_inner - M
    lower_differences = u_candidate_inner - m
    NAD_trouble = np.where(lower_differences < -tolerance, 1, 0)
    NAD_trouble = np.where(upper_differences > tolerance, 1, NAD_trouble)
    NAD_trouble = np.any(NAD_trouble, axis=0, keepdims=True)

    # PAD
    PAD_trouble = np.zeros_like(NAD_trouble, dtype=bool)
    if PAD is not None:
        for var in u_candidate_inner.variable_names:
            PAD_trouble = np.where(
                getattr(u_candidate_inner, var) < PAD[var][0], 1, PAD_trouble
            )
            PAD_trouble = np.where(
                getattr(u_candidate_inner, var) > PAD[var][1], 1, PAD_trouble
            )

    trouble = np.where(PAD_trouble, 1, NAD_trouble)

    return trouble


def broadcase_troubled_cells_to_troubled_interfaces(
    trouble: np.ndarray, xp: str = "numpy"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    args:
        trouble (array_like) : array of troubled cells indicated by 1, shape (# variables, nx, ny, nz)
        xp (str) : 'numpy' or 'cupy'
    returns:
        troubled_x_interfaces (array_like) : array of troubled x interfaces indicated by 1, shape (# variables, nx + 1, ny, nz)
        troubled_y_interfaces (array_like) : array of troubled y interfaces indicated by 1, shape (# variables, nx, ny + 1, nz)
        troubled_z_interfaces (array_like) : array of troubled z interfaces indicated by 1, shape (# variables, nx, ny, nz + 1)
    """
    troubled_x_interfaces = np.zeros(
        (trouble.shape[0], trouble.shape[1] + 1, trouble.shape[2], trouble.shape[3])
    )
    troubled_y_interfaces = np.zeros(
        (trouble.shape[0], trouble.shape[1], trouble.shape[2] + 1, trouble.shape[3])
    )
    troubled_z_interfaces = np.zeros(
        (trouble.shape[0], trouble.shape[1], trouble.shape[2], trouble.shape[3] + 1)
    )
    if xp == "cupy" and CUPY_AVAILABLE:
        troubled_x_interfaces = cp.asarray(troubled_x_interfaces)
        troubled_y_interfaces = cp.asarray(troubled_y_interfaces)
        troubled_z_interfaces = cp.asarray(troubled_z_interfaces)
    troubled_x_interfaces[
        :,
        1:,
        :,
        :,
    ] = trouble
    troubled_x_interfaces[:, :-1, :, :] = np.maximum(
        troubled_x_interfaces[:, :-1, :, :], trouble
    )
    troubled_y_interfaces[:, :, 1:, :] = trouble
    troubled_y_interfaces[:, :, :-1, :] = np.maximum(
        troubled_y_interfaces[:, :, :-1, :], trouble
    )
    troubled_z_interfaces[:, :, :, 1:] = trouble
    troubled_z_interfaces[:, :, :, :-1] = np.maximum(
        troubled_z_interfaces[:, :, :, :-1], trouble
    )
    return troubled_x_interfaces, troubled_y_interfaces, troubled_z_interfaces
