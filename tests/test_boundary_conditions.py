from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.config import conservative_names
from fvhoe.named_array import NamedNumpyArray
import numpy as np
import pytest
from tests.test_utils import meshgen


@pytest.fixture
def sample_array():
    data = np.random.rand(5, 10, 10, 10)
    return NamedNumpyArray(input_array=data, names=conservative_names)


@pytest.mark.parametrize("bc", ["free", "periodic", "symmetric"])
@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_init(bc: str, dim: str):
    """
    Test that the BoundaryCondition class can be initialized in different ways.
    args:
        bc (str) : boundary condition
        dim (str) : dimension
    """
    bc_str = BoundaryCondition(names=conservative_names, **{dim: bc})
    bc_tup = BoundaryCondition(names=conservative_names, **{dim: (bc, bc)})
    bc_dic = BoundaryCondition(
        names=conservative_names, **{dim: {var: bc for var in conservative_names}}
    )
    assert bc_str == bc_tup
    assert bc_str == bc_dic


@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_periodic_symmetric_equivalence(dim: str, N: int = 64, gw: int = 10):
    """
    Test that periodic and symmetric boundary conditions are equivalent for a 3D sinusoidal field.
    args:
        dim (str) : dimension
        N (int) : number of cells
        gw (int) : number of ghost cells
    """
    # create a 3D sinusoidal field
    (X, Y, Z), _ = meshgen((N, N, N))
    sinus_data = np.sin(2 * np.pi * {"x": X, "y": Y, "z": Z}[dim])
    sinus_arr = NamedNumpyArray(
        input_array=np.asarray([sinus_data] * 5), names=conservative_names
    )

    # apply periodic and symmetric boundary conditions
    bc_periodic = BoundaryCondition(names=conservative_names, **{dim: "periodic"})
    bc_symmetric = BoundaryCondition(names=conservative_names, **{dim: "-symmetric"})
    bc_gw = [0, 0, 0]
    bc_gw["xyz".index(dim)] = gw
    sinus_periodic = bc_periodic.apply(sinus_arr, gw=bc_gw)
    sinus_symmetric = bc_symmetric.apply(sinus_arr, gw=bc_gw)

    assert np.max(np.abs(sinus_periodic - sinus_symmetric)) < 1e-15


def test_dirichlet_f_of_xyzt(N: int = 64, gw: int = 10, t: float = 0.5):
    """
    Test that the Dirichlet boundary condition as a function of x, y, z, t is applied correctly for a 3D sinusoidal field.
    args:
        N (int) : number of cells
        gw (int) : number of ghost cells
        t (float) : time
    """

    # create a 3D sinusoidal field
    def f(x, y, z, t):
        return (
            np.sin(2 * np.pi * x)
            * np.sin(2 * np.pi * y)
            * np.sin(2 * np.pi * z)
            * np.sin(2 * np.pi * t)
        )

    h = 1 / N
    (X, Y, Z), _ = meshgen(
        (N + 2 * gw, N + 2 * gw, N + 2 * gw),
        x=(-h * gw, 1 + h * gw),
        y=(-h * gw, 1 + h * gw),
        z=(-h * gw, 1 + h * gw),
    )
    slices = (slice(gw, -gw),) * 3
    data = f(X[slices], Y[slices], Z[slices], t)
    data_all = f(X, Y, Z, t)
    arr = NamedNumpyArray(input_array=np.asarray([data] * 5), names=conservative_names)
    arr_all = NamedNumpyArray(
        input_array=np.asarray([data_all] * 5), names=conservative_names
    )

    # apply dirichlet boundary conditions
    bc = BoundaryCondition(
        names=conservative_names,
        x="dirichlet",
        y="dirichlet",
        z="dirichlet",
        x_value=f,
        y_value=f,
        z_value=f,
        x_domain=(0, 1),
        y_domain=(0, 1),
        z_domain=(0, 1),
        h=(h, h, h),
    )

    assert np.all(bc.apply(arr, gw=(gw, gw, gw), t=t) == arr_all)
