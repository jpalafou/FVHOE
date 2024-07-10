from fvhoe.initial_conditions import empty_primitive
from fvhoe.named_array import NamedNumpyArray
from fvhoe.visualization import sample_circular_average
import numpy as np
import pytest


def primitive_ones(N: int, dims: str) -> NamedNumpyArray:
    """
    args:
        N (int) : mesh size
        dims (str) : contains "x", "y", and/or "z"
    returns:
        out (NamedArray) : primitive array where rho and P are 1 and each velocity component in dims is 1
    """
    shape = (N if "x" in dims else 1, N if "y" in dims else 1, N if "z" in dims else 1)
    out = empty_primitive(shape)
    out.rho = 1
    out.vx = (1 if "x" in dims else 0,)
    out.vy = (1 if "y" in dims else 0,)
    out.vz = (1 if "z" in dims else 0,)
    out.P = 1
    return out


def ones_snapshots(N: int, dims: str) -> list:
    """
    args:
        N (int) : mesh size
        dims (str) : contains "x", "y", and/or "z"
    returns:
        snapshots (list) : snapshot list containing one entry at t=0
    """
    xi = np.linspace(0, 1, N + 1)
    x = 0.5 * (xi[:-1] + xi[1:])
    snapshots = [
        dict(
            t=0.0,
            x=x.copy() if "x" in dims else np.array([0.5]),
            y=x.copy() if "y" in dims else np.array([0.5]),
            z=x.copy() if "z" in dims else np.array([0.5]),
            w=primitive_ones(N, dims=dims),
        )
    ]
    return snapshots


@pytest.mark.parametrize("dims", ["xy", "yz", "zx"])
@pytest.mark.parametrize("param", ["rho", "P", "v"])
@pytest.mark.parametrize(
    "radii", [np.linspace(0, 1, 5), np.linspace(0, 1, 11), np.linspace(0, 1, 21)]
)
def test_sample_circular_average_2D(
    dims: str, param: str, radii: np.ndarray, N: int = 64
):
    """
    test that the circular average of all ones is one and that the average velocity magnitude is sqrt(2)
    args:
        dims (str) : contains "x", "y", and/or "z"
        param (str) : "rho", "P", "v"
        radii (array_like) : interfaces of radial bins
        N (int) : mesh size
    """
    snapshots = ones_snapshots(N=N, dims=dims)
    center = [
        0 if "x" in dims else 0.5,
        0 if "y" in dims else 0.5,
        0 if "z" in dims else 0.5,
    ]
    avg_w, avg_r = sample_circular_average(
        snapshots, param=param, center=center, radii=radii
    )
    if param == "v":
        assert np.max(np.abs(avg_w - np.sqrt(2))) < 1e-15
    else:
        assert np.all(avg_w == 1)


@pytest.mark.parametrize("param", ["rho", "P", "v"])
@pytest.mark.parametrize(
    "radii", [np.linspace(0, 1, 5), np.linspace(0, 1, 11), np.linspace(0, 1, 21)]
)
def test_sample_circular_average_3D(param: str, radii: np.ndarray, N: int = 64):
    """
    test that the circular average of all ones is one and that the average velocity magnitude is sqrt(3)
    args:
        param (str) : "rho", "P", "v"
        radii (array_like) : interfaces of radial bins
        N (int) : mesh size
    """
    snapshots = ones_snapshots(N=N, dims="xyz")
    center = [0, 0, 0]
    avg_w, avg_r = sample_circular_average(
        snapshots, param=param, center=center, radii=radii
    )
    if param == "v":
        assert np.max(np.abs(avg_w - np.sqrt(3))) < 1e-15
    else:
        assert np.all(avg_w == 1)
