from fvhoe.initial_conditions import square, sinus
from fvhoe.fv import fv_average
import numpy as np
import pytest
from tests.test_utils import l1err, meshgen


def test_first_order_cell_average():
    """
    assert first-order finite volume average should is trivial
    """
    (X, Y, Z), h = meshgen((32, 64, 128))

    def f(x, y, z):
        return sinus(x, y, z, dims="xyz", vx=1, vy=2, vz=3)

    assert np.all(f(X, Y, Z) == fv_average(f=f, x=X, y=Y, z=Z, h=h))


@pytest.mark.parametrize("px", [0, 1, 2, 3])
@pytest.mark.parametrize("py", [0, 1, 2, 3])
@pytest.mark.parametrize("pz", [0, 1, 2, 3])
def test_uniform_cell_average(px, py, pz):
    """
    assert finite volume average of a uniform region is trivial
    args:
        px (int) : polynomial interpolation degree in x-direction
        py (int) : polynomial interpolation degree in y-direction
        pz (int) : polynomial interpolation degree in z-direction
    """
    (X, Y, Z), h = meshgen((32, 64, 128))

    def f(x, y, z):
        return square(x, y, z, dims="xyz", vx=1, vy=2, vz=3)

    assert (
        l1err(f(X, Y, Z), fv_average(f=f, x=X, y=Y, z=Z, h=h, p=(px, py, pz))) < 1e-15
    )
