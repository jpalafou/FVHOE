from fvhoe.initial_conditions import square, sinus
from fvhoe.fv import (
    interpolate_cell_centers,
    interpolate_fv_averages,
    fv_average,
    fv_uniform_meshgen,
)
import numpy as np
import pytest
from tests.utils import l1err


def test_first_order_cell_average():
    """
    assert first-order finite volume average should is trivial
    """
    X, Y, Z = fv_uniform_meshgen((32, 64, 128))
    h = (1 / 32, 1 / 64, 1 / 128)
    f = sinus(dims="xyz", vx=1, vy=2, vz=3)

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
    X, Y, Z = fv_uniform_meshgen((32, 64, 128))
    h = (1 / 32, 1 / 64, 1 / 128)
    f = square(dims="xyz", vx=1, vy=2, vz=3)

    assert (
        l1err(f(X, Y, Z), fv_average(f=f, x=X, y=Y, z=Z, h=h, p=(px, py, pz))) < 1e-15
    )


@pytest.mark.parametrize(
    "transformation", [interpolate_cell_centers, interpolate_fv_averages]
)
@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 7, 8])
def test_interpolation_symmetry(transformation: callable, p: int, N: int = 128):
    """
    assert finite volume cell center interpolation is symmetric about x=0, y=0, z=0
    args:
        transformation (callable) : interpolation function (fv to cell centers or visa versa)
        p (int) : polynomial interpolation degree
        N (int) : number of cells
    """
    X, Y, Z = fv_uniform_meshgen((N, N, N))
    R = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
    data1 = np.where(R < 0.25, 1, 0).astype(float)[np.newaxis, ...]
    data2 = transformation(data1, p=(p, p, p))
    M = N - 2 * (-(-p // 2))
    assert l1err(data2[:, : M // 2, :, :], data2[:, : M // 2 - 1 : -1, :, :]) < 1e-15
    assert l1err(data2[:, :, : M // 2, :], data2[:, :, : M // 2 - 1 : -1, :]) < 1e-15
    assert l1err(data2[:, :, :, : M // 2], data2[:, :, :, : M // 2 - 1 : -1]) < 1e-15
    assert (
        l1err(data2[:, : M // 2, :, :], np.swapaxes(data2[:, :, : M // 2, :], 1, 2))
        < 1e-15
    )
    assert (
        l1err(data2[:, : M // 2, :, :], np.swapaxes(data2[:, :, :, : M // 2], 1, 3))
        < 1e-15
    )
