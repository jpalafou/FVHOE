from fvhoe.hydro import HydroState
from fvhoe.visualization import sample_circular_average
import numpy as np
import pytest

_hs = HydroState(ndim=1)


def _primitive_ones(N: int, dims: str) -> np.ndarray:
    """
    args:
        N (int) : mesh size
        dims (str) : contains "x", "y", and/or "z"
    returns:
        out (np.ndarray) : primitive array where rho and P are 1 and each velocity component in dims is 1
    """
    shape = (N if "x" in dims else 1, N if "y" in dims else 1, N if "z" in dims else 1)

    out = np.empty((5, *shape))
    out[_hs("rho")] = 1.0
    out[_hs("vx")] = 1.0 if "x" in dims else 0.0
    out[_hs("vy")] = 1.0 if "y" in dims else 0.0
    out[_hs("vz")] = 1.0 if "z" in dims else 0.0
    out[_hs("P")] = 1.0
    return out


def _ones_snapshots(N: int, dims: str) -> list:
    """
    args:
        N (int) : mesh size
        dims (str) : contains "x", "y", and/or "z"
    returns:
        snapshots (list) : snapshot list containing one entry at t=0
    """
    xi = np.linspace(0, 1, N + 1)
    x = (0.5 * (xi[:-1] + xi[1:])).reshape(-1, 1, 1)
    snapshots = [
        dict(
            t=0.0,
            x=x.copy() if "x" in dims else np.array([[[0.5]]]),
            y=x.copy() if "y" in dims else np.array([[[0.5]]]),
            z=x.copy() if "z" in dims else np.array([[[0.5]]]),
            w=_primitive_ones(N, dims=dims),
        )
    ]
    return snapshots


@pytest.fixture
def mock_EulerSolver():
    """Fixture for an EulerSolver with all ones and zeros for snapshot data."""

    class MockEulerSolver:
        def __init__(self, N: int, dims: str):
            self.snapshots = _ones_snapshots(N=N, dims=dims)
            self.hydro_state = HydroState()

    return MockEulerSolver


@pytest.mark.parametrize("dims", ["xy", "yz", "zx"])
@pytest.mark.parametrize("param", ["rho", "P", "v"])
@pytest.mark.parametrize(
    "radii", [np.linspace(0, 1, 5), np.linspace(0, 1, 11), np.linspace(0, 1, 21)]
)
def test_sample_circular_average_2D(
    mock_EulerSolver, dims: str, param: str, radii: np.ndarray, N: int = 64
):
    """
    test that the circular average of all ones is one and that the average velocity magnitude is sqrt(2)
    args:
        mock_EulerSolver (class) : mock class for EulerSolver
        dims (str) : contains "x", "y", and/or "z"
        param (str) : "rho", "P", "v"
        radii (array_like) : interfaces of radial bins
        N (int) : mesh size
    """
    mock_solver = mock_EulerSolver(N=N, dims=dims)
    center = [
        0 if "x" in dims else 0.5,
        0 if "y" in dims else 0.5,
        0 if "z" in dims else 0.5,
    ]
    avg_w, avg_r = sample_circular_average(
        mock_solver, param=param, center=center, radii=radii
    )
    if param == "v":
        assert np.max(np.abs(avg_w - np.sqrt(2))) < 1e-15
    else:
        assert np.all(avg_w == 1)


@pytest.mark.parametrize("param", ["rho", "P", "v"])
@pytest.mark.parametrize(
    "radii", [np.linspace(0, 1, 5), np.linspace(0, 1, 11), np.linspace(0, 1, 21)]
)
def test_sample_circular_average_3D(
    mock_EulerSolver, param: str, radii: np.ndarray, N: int = 64
):
    """
    test that the circular average of all ones is one and that the average velocity magnitude is sqrt(3)
    args:
        mock_EulerSolver (class) : mock class for EulerSolver
        param (str) : "rho", "P", "v"
        radii (array_like) : interfaces of radial bins
        N (int) : mesh size
    """
    mock_solver = mock_EulerSolver(N=N, dims="xyz")
    center = [0, 0, 0]
    avg_w, avg_r = sample_circular_average(
        mock_solver, param=param, center=center, radii=radii
    )
    if param == "v":
        assert np.max(np.abs(avg_w - np.sqrt(3))) < 1e-15
    else:
        assert np.all(avg_w == 1)
