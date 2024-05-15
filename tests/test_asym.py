from functools import partial
from fvhoe.fv import get_window
import numpy as np
import pytest


def interpolate_right_face(u, axis):
    gw = partial(get_window, ndim=u.ndim, axis=axis)
    out = (1 / 4) * (
        -1 * u[gw(cut=(0, 2))] + 4 * u[gw(cut=(1, 1))] + 1 * u[gw(cut=(2, 0))]
    )
    return out


@pytest.mark.parametrize("test_number", range(1))
@pytest.mark.parametrize("compared_axes", [(1, 2), (2, 3)])
def test_stencil_1d(test_number, compared_axes, N=256, n_steps=1000):
    x = np.linspace(0, 1, N + 1)
    x = 0.5 * (x[1:] + x[:-1])
    data = np.where(np.logical_and(x > 0.25, x < 0.75), 2, 1)
    data = np.asarray([0.8 * data, 0.9 * data, data, 1.1 * data, 1.2 * data])
    ax1, ax2 = compared_axes
    gw1 = partial(get_window, ndim=4, axis=ax1)
    gw2 = partial(get_window, ndim=4, axis=ax2)

    # initialize first array
    shape1 = [5, 1, 1, 1]
    shape1[ax1] = N
    u1 = np.empty(tuple(shape1))

    # assign data and pad widths
    slice1 = [slice(None), 0, 0, 0]
    slice1[ax1] = slice(None)
    u1[tuple(slice1)] = data.copy()
    u1_pad_width = [(0, 0)] * 4
    u1_pad_width[ax1] = (1, 1)

    # initialize second array
    shape2 = [5, 1, 1, 1]
    shape2[ax2] = N
    u2 = np.empty(tuple(shape2))

    # assign data and pad widths
    slice2 = [slice(None), 0, 0, 0]
    slice2[ax2] = slice(None)
    u2[tuple(slice2)] = data.copy()
    u2_pad_width = [(0, 0)] * 4
    u2_pad_width[ax2] = (1, 1)

    # advection
    v = 1
    C = 0.8
    h = np.mean(x[1:] - x[:-1])
    dt = C * h / v

    for _ in range(n_steps):
        u1_gw = np.pad(u1, pad_width=u1_pad_width, mode="wrap")
        first_order_fluxes = v * u1_gw[gw1(cut=(0, 1))]
        u1 = u1 - (dt / h) * (
            first_order_fluxes[gw1(cut=(1, 0))] - first_order_fluxes[gw1(cut=(0, 1))]
        )

    for _ in range(n_steps):
        u2_gw = np.pad(u2, pad_width=u2_pad_width, mode="wrap")
        first_order_fluxes = v * u2_gw[gw2(cut=(0, 1))]
        u2 = u2 - (dt / h) * (
            first_order_fluxes[gw2(cut=(1, 0))] - first_order_fluxes[gw2(cut=(0, 1))]
        )

    assert np.mean(np.abs(u1[tuple(slice1)] - u2[tuple(slice2)])) == 0
