from fvhoe.initial_conditions import InitialCondition
import numpy as np
import pytest
from typing import Callable

# Assuming InitialCondition and the necessary HydroState class structure are imported


class SimpleIC(InitialCondition):
    """Concrete InitialCondition subclass for testing purposes."""

    def base_ic(
        self, *args, **kwargs
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        # Simple base_ic returning a constant array
        def core_ic_func(x, y, z):
            return np.ones((5, *x.shape))

        return core_ic_func


@pytest.fixture
def mock_hydro_state():
    """Fixture for a HydroState with and without passive scalars."""

    class MockHydroState:
        def __init__(self, passives=None):
            self.passive_scalars = passives if passives else ()
            self.includes_passives = bool(self.passive_scalars)
            self.variable_map = {
                name: idx + 5 for idx, name in enumerate(self.passive_scalars)
            }

    return MockHydroState


@pytest.fixture
def sample_coords():
    """Sample coordinates for x, y, z to be used as input."""
    nx, ny, nz = 10, 10, 10
    x, y, z = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
    return x, y, z


def test_initial_condition_no_passives(mock_hydro_state, sample_coords):
    """Test IC generation without any passive scalars."""
    hs = mock_hydro_state()  # No passive scalars
    ic_instance = SimpleIC()

    ic_func = ic_instance.build_ic(hs)
    x, y, z = sample_coords

    result = ic_func(x, y, z)
    assert result.shape == (
        5,
        *x.shape,
    ), "Output shape should be (5, nx, ny, nz) for no passive scalars."
    assert np.all(result[:5] == 1), "Base IC values should be set to ones."


def test_initial_condition_with_passives(mock_hydro_state, sample_coords):
    """Test IC generation with passive scalars provided."""
    hs = mock_hydro_state(passives=["tracer1", "tracer2"])
    print(hs.variable_map)

    passive_ic_funcs = {
        "tracer1": lambda x, y, z: np.full(x.shape, 2),
        "tracer2": lambda x, y, z: np.full(x.shape, 3),
    }
    ic_instance = SimpleIC()
    ic_func = ic_instance.build_ic(hs, passive_ic_funcs)
    x, y, z = sample_coords

    result = ic_func(x, y, z)
    assert result.shape == (
        7,
        *x.shape,
    ), "Output shape should be (5 + 2, nx, ny, nz) with two passive scalars."
    assert np.all(result[:5] == 1), "Base IC values should be ones."
    assert np.all(result[5] == 2), "Tracer1 should be set to twos."
    assert np.all(result[6] == 3), "Tracer2 should be set to threes."


def test_missing_passive_scalar_raises_error(mock_hydro_state, sample_coords):
    """Test that an error is raised if a required passive scalar IC is missing."""
    hs = mock_hydro_state(passives=["tracer1", "tracer2"])
    passive_ic_funcs = {
        "tracer1": lambda x, y, z: np.full(x.shape, 2),
        # Missing tracer2
    }
    ic_instance = SimpleIC()

    with pytest.raises(ValueError, match="Missing IC definitions for passive scalars"):
        ic_instance.build_ic(hs, passive_ic_funcs)
