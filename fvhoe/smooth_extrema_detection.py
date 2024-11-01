from fvhoe.hydro import HydroState
from fvhoe.fv import second_order_central_difference, get_symmetric_slices
import numpy as np

_hs = HydroState(ndim=4)


def compute_1d_smooth_extrema_detector(
    u: np.ndarray, dim: str, eps: float = 1e-10
) -> np.ndarray:
    """
    compute smooth extrema detector alpha along specified direction
    args:
        u (array_like) : array of shape (# vars, nx, ny, nz)
        dim (str) : "x", "y", "z"
        eps (float) : how close to 0 dv is permitted to reach
    returns:
        out (array_like) : first neighbor minimum of alpha along specified direction (shorter by 6 elements)
    """

    # define slicing function
    axis = {"x": 1, "y": 2, "z": 3}[dim]
    slices = get_symmetric_slices(u.ndim, 3, axis)

    # get slopes
    dU = second_order_central_difference(u, axis)
    d2U = second_order_central_difference(dU, axis)
    dv = 0.5 * d2U
    dv[...] = np.where(
        np.abs(dv) <= eps, np.where(dv >= 0, eps, -eps), dv
    )  # avoid dividing by 0

    # left detector
    vL = dU[slices[0]] - dU[slices[1]]
    alphaL = -np.where(dv < 0, np.where(vL > 0, vL, 0), np.where(vL < 0, vL, 0)) / dv
    alphaL[...] = (
        -np.where(dv < 0, np.where(vL > 0, vL, 0), np.where(vL < 0, vL, 0)) / dv
    )
    alphaL[...] = np.where(alphaL < 1, alphaL, 1)

    # right detector
    vR = dU[slices[2]] - dU[slices[1]]
    alphaR = np.where(dv > 0, np.where(vR > 0, vR, 0), np.where(vR < 0, vR, 0)) / dv
    alphaR[...] = np.where(np.abs(dv) <= eps, 1, alphaR)
    alphaR[...] = np.where(alphaR < alphaL, alphaR, alphaL)

    # take local minimum
    alpha = np.minimum(alphaL, alphaR)
    out = np.minimum(np.minimum(alpha[slices[2]], alpha[slices[1]]), alpha[slices[0]])
    return out


def compute_2d_smooth_extrema_detector(
    u: np.ndarray, dims: str, eps: float = 1e-10
) -> np.ndarray:
    """
    compute smooth extrema detector alpha in x and y directions
        args:
            u (array_like) : array of shape (# vars, nx, ny, nz)
            dims (str) : two characters sampling ["x", "y", "z"] without replacement
            eps (float) : how close to 0 dv is permitted to reach
        returns:
            out (array_like) : first neighbor minimum of alpha along each specified direction (shorter by 6 elements)
    """
    axis1 = {"x": 1, "y": 2, "z": 3}[dims[0]]
    axis2 = {"x": 1, "y": 2, "z": 3}[dims[1]]
    alpha_dim1 = compute_1d_smooth_extrema_detector(
        u[_hs(axis=axis2, cut=(3, -3))], dim=dims[0], eps=eps
    )
    alpha_dim2 = compute_1d_smooth_extrema_detector(
        u[_hs(axis=axis1, cut=(3, -3))], dim=dims[1], eps=eps
    )
    out = np.minimum(alpha_dim1, alpha_dim2)
    return out


def compute_3d_smooth_extrema_detector(u: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    compute smooth extrema detector alpha in x, y, and z directions
        args:
            u (array_like) : array of shape (# vars, nx, ny, nz)
            eps (float) : how close to 0 dv is permitted to reach
        returns:
            out (array_like) : first neighbor minimum of alpha along each specified direction (shorter by 6 elements)
    """
    alpha_xy = compute_2d_smooth_extrema_detector(
        u[_hs(axis=3, cut=(3, -3))], dims="xy", eps=eps
    )
    alpha_z = compute_1d_smooth_extrema_detector(
        u[_hs(axis=1, cut=(3, -3))][_hs(axis=2, cut=(3, -3))],
        dim="z",
        eps=eps,
    )
    out = np.minimum(alpha_xy, alpha_z)
    return out
