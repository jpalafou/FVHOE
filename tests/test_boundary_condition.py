from fvhoe.boundary_conditions import fd, set_finite_difference_bc
from fvhoe.fv import get_window
from tests.test_utils import meshgen
import numpy as np
import pytest


@pytest.mark.parametrize("test_number", range(10))
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("pos", ["l", "r"])
def test_neumann_boundary(test_number, ndim, p, pos):
    """
    assert boundaries can be set to satisfy neumann gradient
    args:
        test_number (int) : arbitrary test label
        ndim (int) : number of dimensions
        p (int) : polynomial degree
        pos (str) "l" or "r"
    """
    # random variables
    axis = np.random.randint(ndim)
    ng = max(int(np.ceil(p / 2)) + 1, 2 * int(np.ceil(p / 2)))

    # initialize domain
    u_shape = {1: (32, 1, 1), 2: (32, 32, 1), 3: (32, 32, 32)}[ndim]
    (X, Y, Z), _ = meshgen(u_shape)
    u = 0.5 * np.cos(2 * np.pi * (X + Y + Z)) - 0.5
    slope = 0

    # create boundary zones
    pad_width = [(0, 0)] * 3
    pad_width[axis] = (ng, ng)
    u = np.pad(u, pad_width=pad_width, constant_values=np.nan)

    # set boundaries
    set_finite_difference_bc(u=u, p=p, h=1, axis=axis, pos=pos, slope=slope, ng=ng)

    # check derivatives
    u_derivative = fd(u=u, p=p, h=1, axis=axis)

    if pos == "l":
        boundary_slopes = u_derivative[get_window(ndim=ndim, axis=axis, cut=(0, -ng))]
    else:
        boundary_slopes = u_derivative[get_window(ndim=ndim, axis=axis, cut=(-ng, 0))]

    assert np.mean(np.abs(boundary_slopes - slope)) < 1e-15
