from fvhoe.hydro import compute_conservatives, compute_primitives
import numpy as np
import pytest
from tests.utils import mse
from typing import Tuple


@pytest.mark.parametrize("test_number", range(5))
@pytest.mark.parametrize(
    "f1_f2",
    [
        (compute_primitives, compute_conservatives),
        (compute_conservatives, compute_primitives),
    ],
)
def test_transformation(
    test_number: int, f1_f2: Tuple[callable, callable], gamma: float = 5 / 3
):
    """
    assert invertibility of transformatons betwen primitive and conservative variables
    args:
        test_number (int) : arbitrary test label
        f1_f2 (Tuple[callable, callable]) : transformation functions like
            compute_conservatives and compute_primitives
        gamma (float) : specific heat ratio
    """
    f1, f2 = f1_f2
    u = 2 * np.random.rand(5, 64, 64, 64) + 1
    assert mse(u, f1(f2(u, gamma=gamma), gamma=gamma)) < 1e-15
