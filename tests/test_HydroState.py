from fvhoe.hydro import HydroState
import numpy as np
import pytest


def test_hydrostate_init_no_scalars():
    # Test initialization without passive scalars
    state = HydroState(ndim=4)

    # Check basic variable map
    expected_map = {
        "rho": 0,
        "vx": 1,
        "mx": 1,
        "vy": 2,
        "my": 2,
        "vz": 3,
        "mz": 3,
        "v": np.arange(1, 4),
        "m": np.arange(1, 4),
        "P": 4,
        "E": 4,
    }

    assert set(expected_map.keys()) == set(state.variable_map.keys())
    for key in expected_map.keys():
        if isinstance(expected_map[key], np.ndarray):
            assert np.array_equal(expected_map[key], state.variable_map[key])
        else:
            assert expected_map[key] == state.variable_map[key]
    assert not state.includes_passives


def test_hydrostate_init_with_scalars():
    # Test initialization with passive scalars
    scalars = ("tracer1", "tracer2")
    state = HydroState(passive_scalars=scalars, ndim=4)

    # Check the variable map for scalars
    expected_map = {
        "rho": 0,
        "vx": 1,
        "mx": 1,
        "vy": 2,
        "my": 2,
        "vz": 3,
        "mz": 3,
        "v": np.arange(1, 4),
        "m": np.arange(1, 4),
        "P": 4,
        "E": 4,
        "tracer1": 5,
        "tracer2": 6,
        "passive_scalars": np.arange(5, 7),
    }

    assert set(expected_map.keys()) == set(state.variable_map.keys())
    for key in expected_map.keys():
        if isinstance(expected_map[key], np.ndarray):
            assert np.array_equal(expected_map[key], state.variable_map[key])
        else:
            assert expected_map[key] == state.variable_map[key]
    assert state.includes_passives


def test_single_variable_slice():
    # Test slicing with a single variable
    state = HydroState(ndim=4)

    # Test for 'rho' variable
    result = state(var="rho")
    expected = (0, slice(None), slice(None), slice(None))

    assert result == expected


def test_multiple_variable_slice():
    # Test slicing with multiple variables
    state = HydroState(ndim=4)

    # Test for 'vx', 'vy', 'vz'
    result = state(var=("vx", "vy", "vz"))
    expected = (np.array([1, 2, 3]), slice(None), slice(None), slice(None))

    assert np.array_equal(result[0], expected[0])
    assert result[1:] == expected[1:]


def test_coordinate_slicing():
    # Test coordinate slicing for x, y, and z
    state = HydroState(ndim=4)

    # Slicing for x = (1, 3), y = (2, 4), z = (0, 2)
    result = state(x=(1, 3), y=(2, 4), z=(0, 2))
    expected = (slice(None), slice(1, 3), slice(2, 4), slice(None, 2))

    assert result == expected


def test_invalid_variable_name():
    # Test error handling for invalid variable name
    state = HydroState(ndim=4)

    with pytest.raises(ValueError, match="Variable 'invalid_var' not found."):
        state(var="invalid_var")


def test_invalid_axis_slice():
    # Test error handling for invalid axis slicing
    state = HydroState(ndim=4)

    # Invalid axis (out of ndim range)
    with pytest.raises(ValueError, match="Invalid axis 4 for array with 4 dimensions."):
        state(axis=4, cut=(0, 2))


def test_invalid_slice_tuple_length():
    # Test error handling for invalid slice tuple length
    state = HydroState(ndim=4)

    # Invalid tuple length for x slice
    with pytest.raises(ValueError, match="Invalid tuple length for axis 1: 3"):
        state(x=(0, 1, 2))


def test_invalid_slice_type():
    # Test error handling for invalid slice type (non-tuple)
    state = HydroState(ndim=4)

    # Invalid slice type for y
    with pytest.raises(ValueError, match="Expected a tuple"):
        state(y=5)


def test_passive_scalars_slice():
    # Test slicing that includes passive scalars
    state = HydroState(passive_scalars=("scalar1", "scalar2", "scalar3"), ndim=4)

    # Test for passive scalar "scalar1"
    result = state(var="passive_scalars")
    expected = (np.array([5, 6, 7]), slice(None), slice(None), slice(None))

    assert np.array_equal(result[0], expected[0])
    assert result[1:] == expected[1:]
