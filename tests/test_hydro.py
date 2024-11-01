from fvhoe.hydro import compute_conservatives, compute_primitives, HydroState
import numpy as np
import pytest
from tests.utils import l1err


@pytest.fixture
def hs():
    return HydroState(ndim=1, passive_scalars=("passive_scalar1", "passive_scalar2"))


@pytest.mark.parametrize("test_number", range(5))
def test_conservative_to_primitive_invertibility(
    hs: HydroState, test_number: int, gamma: float = 5 / 3
):
    """
    assert invertibility of transformations between conservative and primitive variables
    args:
        hs (HydroState) : HydroState object
        test_number (int) : arbitrary test label
        gamma (float) : specific heat ratio
    """
    u1 = 2 * np.random.rand(7, 64, 64, 64) + 1
    u2 = compute_conservatives(hs, compute_primitives(hs, u1, gamma=gamma), gamma=gamma)
    assert l1err(u1[hs("E")], u2[hs("E")]) < 1e-15


@pytest.mark.parametrize("test_number", range(5))
def test_primitive_to_conservative_invertibility(
    hs: HydroState, test_number: int, gamma: float = 5 / 3
):
    """
    assert invertibility of transformations between primitive and conservative variables
    args:
        hs (HydroState) : HydroState object
        test_number (int) : arbitrary test label
        gamma (float) : specific heat ratio
    """
    w1 = 2 * np.random.rand(7, 64, 64, 64) + 1
    w2 = compute_primitives(hs, compute_conservatives(hs, w1, gamma=gamma), gamma=gamma)
    assert l1err(w1[hs("P")], w2[hs("P")]) < 1e-15
