from fvhoe.array_manager import get_array_slice as slc
from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.fv import fv_uniform_meshgen
from fvhoe.initial_conditions import variable_array, shu_osher_1d
from fvhoe.solver import EulerSolver
import numpy as np
import pytest
from tests.test_utils import l1err


@pytest.fixture
def sample_array():
    return np.random.rand(5, 10, 10, 10)


@pytest.mark.parametrize("bc", ["free", "outflow", "periodic", "reflective"])
@pytest.mark.parametrize("dim", "xyz")
def test_init(bc: str, dim: str):
    """
    Test that the BoundaryCondition class can be initialized in different ways.
    args:
        bc (str) : boundary condition
        dim (str) : dimension
    """
    bc_str = BoundaryCondition(**{dim: bc})
    bc_tup = BoundaryCondition(**{dim: (bc, bc)})
    assert getattr(bc_str, dim) == getattr(bc_tup, dim)


@pytest.mark.parametrize("empty_dim", "xyz")
@pytest.mark.parametrize("bc", ["free", "outflow", "periodic", "reflective"])
def test_2d_bc(empty_dim: str, bc: str, N: int = 64, gw: int = 10):
    """
    Test that the boundary condition applied to a third dimension with gw=0 is inconsequential.
    args:
        empty_dim (str) : dimension with gw=0
        N (int) : number of cells
        gw (int) : number of ghost cells
    """
    # create a 2D random field
    arr = np.random.rand(
        5, *{"x": (1, N, N), "y": (N, 1, N), "z": (N, N, 1)}[empty_dim]
    )
    gws = [
        0 if empty_dim == "x" else gw,
        0 if empty_dim == "y" else gw,
        0 if empty_dim == "z" else gw,
    ]

    # define different boundary conditions
    trivial_bc = BoundaryCondition(
        x=None if empty_dim == "x" else bc,
        y=None if empty_dim == "y" else bc,
        z=None if empty_dim == "z" else bc,
    )
    nontrivial_bc = BoundaryCondition(x=bc, y=bc, z=bc)

    # apply boundary conditions
    arr1 = trivial_bc.apply(arr, gw=gws)
    arr2 = nontrivial_bc.apply(arr, gw=gws)

    # check that the results are the same
    assert np.all(arr1 == arr2)


@pytest.mark.parametrize("dim", "xyz")
def test_periodic_symmetric_equivalence(dim: str, N: int = 64, gw: int = 10):
    """
    Test that periodic and symmetric boundary conditions are equivalent for a 3D sinusoidal field.
    args:
        dim (str) : dimension
        N (int) : number of cells
        gw (int) : number of ghost cells
    """
    # create a 3D sinusoidal field
    (X, Y, Z) = fv_uniform_meshgen((N, N, N))
    sinus_arr = np.array(5 * [np.cos(2 * np.pi * {"x": X, "y": Y, "z": Z}[dim])])

    # apply periodic and symmetric boundary conditions
    bc_periodic = BoundaryCondition()
    bc_symmetric = BoundaryCondition(**{dim: "outflow"})
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
        data = (
            np.sin(2 * np.pi * x)
            * np.sin(2 * np.pi * y)
            * np.sin(2 * np.pi * z)
            * np.sin(2 * np.pi * t)
        )
        out = variable_array(
            shape=x.shape,
            rho=data,
            P=2 * data,
            vx=3 * data,
            vy=4 * data,
            vz=5 * data,
            conservative=True,
        )
        return out

    h = 1 / N
    X, Y, Z = fv_uniform_meshgen(
        (N + 2 * gw, N + 2 * gw, N + 2 * gw),
        x=(-h * gw, 1 + h * gw),
        y=(-h * gw, 1 + h * gw),
        z=(-h * gw, 1 + h * gw),
    )
    arr_all = f(X, Y, Z, t)

    (X, Y, Z), slab_coords = fv_uniform_meshgen(
        (N, N, N),
        slab_thickness=(gw, gw, gw),
    )
    arr = f(X, Y, Z, t)

    # apply dirichlet boundary conditions
    bc = BoundaryCondition(
        x="dirichlet",
        y="dirichlet",
        z="dirichlet",
        x_value=f,
        y_value=f,
        z_value=f,
        slab_coords=slab_coords,
    )

    assert np.all(bc.apply(arr, gw=(gw, gw, gw), t=t) == arr_all)


@pytest.mark.parametrize("dim", ["x", "y", "z"])
@pytest.mark.parametrize("p", [0, 1, 2, 3, 8])
def test_ic_bc(dim: str, p: int, N: int = 100, gamma: float = 1.4, T: float = 1.8):
    """
    Test that boundary condition type "ic" is equivalent to "dirichlet" for the Shu Osher problem.
    args:
        dim (str) : dimension
        p (int) : polynomial degree
        N (int) : number of cells
        gamma (float) : specific heat ratio
        T (float) : final time
    """
    # set up solvers
    solver_configs = dict(
        w0=shu_osher_1d(dim=dim),
        x=(0, 10) if dim == "x" else (0, 1),
        y=(0, 10) if dim == "y" else (0, 1),
        z=(0, 10) if dim == "z" else (0, 1),
        nx=N if dim == "x" else 1,
        ny=N if dim == "y" else 1,
        nz=N if dim == "z" else 1,
        px=p if dim == "x" else 0,
        py=p if dim == "y" else 0,
        pz=p if dim == "z" else 0,
        riemann_solver="hllc",
        gamma=gamma,
        a_posteriori_slope_limiting=p > 0,
        slope_limiter="minmod",
    )

    # dirichlet bc
    dirichlet_arrs = (
        np.array(
            [
                3.857143,
                10.33333 / (gamma - 1) + 0.5 * 3.857143 * 2.629369 * 2.629369,
                3.857143 * 2.629369 if dim == "x" else 0,
                3.857143 * 2.629369 if dim == "y" else 0,
                3.857143 * 2.629369 if dim == "z" else 0,
            ]
        ),
        np.array([1 + 0.2 * np.sin(5 * 5), 1 / (gamma - 1), 0, 0, 0]),
    )

    # solver with dirichlet bc
    solver_dirichlet = EulerSolver(
        bc=BoundaryCondition(
            x="dirichlet" if dim == "x" else None,
            y="dirichlet" if dim == "y" else None,
            z="dirichlet" if dim == "z" else None,
            x_value=dirichlet_arrs if dim == "x" else None,
            y_value=dirichlet_arrs if dim == "y" else None,
            z_value=dirichlet_arrs if dim == "z" else None,
        ),
        **solver_configs,
    )

    # solver with ic bc
    solver_ic = EulerSolver(
        bc=BoundaryCondition(
            x="ic" if dim == "x" else None,
            y="ic" if dim == "y" else None,
            z="ic" if dim == "z" else None,
        ),
        **solver_configs,
    )

    # run solvers
    solver_dirichlet.rkorder(T)
    solver_ic.rkorder(T)

    # compare solvers
    assert (
        l1err(
            solver_dirichlet.snapshots[-1]["w"][slc("P")],
            solver_ic.snapshots[-1]["w"][slc("P")],
        )
        < 1e-14
    )
