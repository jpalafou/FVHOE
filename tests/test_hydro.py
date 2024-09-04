from fvhoe.array_manager import get_array_slice as slc
from fvhoe.hydro import compute_conservatives, compute_primitives
import numpy as np
import pytest
from tests.test_utils import l1err


@pytest.mark.parametrize("test_number", range(5))
def test_conservative_to_primitive_invertibility(
    test_number: int, gamma: float = 5 / 3
):
    """
    assert invertibility of transformations between conservative and primitive variables
    args:
        test_number (int) : arbitrary test label
        gamma (float) : specific heat ratio
    """
    u1 = 2 * np.random.rand(5, 64, 64, 64) + 1
    u2 = compute_conservatives(compute_primitives(u1, gamma=gamma), gamma=gamma)
    assert l1err(u1[slc("E")], u2[slc("E")]) < 1e-15


@pytest.mark.parametrize("test_number", range(5))
def test_primitive_to_conservative_invertibility(
    test_number: int, gamma: float = 5 / 3
):
    """
    assert invertibility of transformations between primitive and conservative variables
    args:
        test_number (int) : arbitrary test label
        gamma (float) : specific heat ratio
    """
    w1 = 2 * np.random.rand(5, 64, 64, 64) + 1
    w2 = compute_primitives(compute_conservatives(w1, gamma=gamma), gamma=gamma)
    assert l1err(w1[slc("P")], w2[slc("P")]) < 1e-15
