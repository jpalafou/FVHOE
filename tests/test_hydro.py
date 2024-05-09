from fvhoe.hydro import compute_conservatives, compute_primitives
import numpy as np
import pytest
from tests.utils import mse


@pytest.mark.parametrize("test_number", range(5))
@pytest.mark.parametrize(
    "f1_f2",
    [
        (compute_primitives, compute_conservatives),
        (compute_conservatives, compute_primitives),
    ],
)
def test_transformation(test_number, f1_f2, gamma=5 / 3):
    """
    Testing invertibility of transformatons betwen primitive and conservative variables
    """
    f1, f2 = f1_f2
    u = 2 * np.random.rand(5, 64, 64, 64) + 1
    assert mse(u, f1(f2(u, gamma=gamma), gamma=gamma)) < 1e-15
