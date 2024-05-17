from fvhoe.config import conservative_names, primitive_names
from fvhoe.hydro import compute_conservatives, compute_primitives
from fvhoe.named_array import NamedNumpyArray
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
    data = 2 * np.random.rand(5, 64, 64, 64) + 1
    u1 = NamedNumpyArray(data, conservative_names)
    u2 = compute_conservatives(compute_primitives(u1, gamma=gamma), gamma=gamma)
    assert l1err(u1.E, u2.E) < 1e-15


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
    data = 2 * np.random.rand(5, 64, 64, 64) + 1
    w1 = NamedNumpyArray(data, primitive_names)
    w2 = compute_primitives(compute_conservatives(w1, gamma=gamma), gamma=gamma)
    assert l1err(w1.P, w2.P) < 1e-15
