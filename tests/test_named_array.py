from fvhoe.named_array import NamedCupyArray, NamedNumpyArray
import numpy as np
import os
import pytest
from typing import Iterable, Union


@pytest.mark.parametrize("test_number", range(10))
@pytest.mark.parametrize("NamedArray", [NamedCupyArray, NamedNumpyArray])
def test_name_permutations(test_number: int, NamedArray: callable):
    """
    assert that the order of the names list does not affect the value of the data
    args:
        test_number (int) : arbitrary test identifier
        NamedArray (callable) : NamedNumpyArray or NamedCupyArray
    """
    data = np.random.rand(10, 10)

    # a is first axis, b is second axis
    named_arr = NamedArray(
        np.empty((2, 10, 10)),
        [
            "a",
            "b",
        ],
    )
    named_arr.a = data + 1
    named_arr.b = data + 2

    # b is first axis, a is second axis
    permuted_named_arr = NamedArray(np.empty((2, 10, 10)), ["b", "a"])
    permuted_named_arr.a = data + 1
    permuted_named_arr.b = data + 2

    assert np.all(named_arr.a == permuted_named_arr.a)
    assert np.all(named_arr.b == permuted_named_arr.b)


@pytest.mark.parametrize("NamedArray", [NamedCupyArray, NamedNumpyArray])
def test_np_sqrt(NamedArray: callable):
    """
    assert that numpy functions work on named arrays
    args:
        NamedArray (callable) : NamedNumpyArray or NamedCupyArray
    """
    named_arr = NamedArray(
        100 * np.ones((2, 10, 10), dtype=int),
        [
            "a",
            "b",
        ],
    )
    assert np.all(np.sqrt(named_arr) == np.sqrt(100))


@pytest.mark.parametrize("NamedArray", [NamedCupyArray, NamedNumpyArray])
def test_partial_assignment(NamedArray: callable):
    """
    assert that the sum of a 1, -1 array and a -1, 1 array is 0
    args:
        NamedArray (callable) : NamedNumpyArray or NamedCupyArray
    """
    named_arr1 = NamedArray(
        np.ones((2, 10, 10)),
        [
            "a",
            "b",
        ],
    )
    named_arr1.a = -1
    named_arr2 = NamedArray(
        np.ones((2, 10, 10)),
        [
            "a",
            "b",
        ],
    )
    named_arr2.b = -1
    assert np.all(named_arr1 + named_arr2 == 0)


@pytest.mark.parametrize("NamedArray", [NamedCupyArray, NamedNumpyArray])
def test_shared_memory(NamedArray: callable):
    """
    assert variable attribute and corresponding array index share memory
    args:
        NamedArray (callable) : NamedNumpyArray or NamedCupyArray
    """
    names = ["a", "b", "c"]
    named_arr = NamedArray(np.empty((len(names), 10, 10)), names)

    for i, name in enumerate(names):
        assert np.may_share_memory(named_arr[i, ...], named_arr.__getattr__(name))
        for j in range(len(names)):
            if i == j:
                continue
            assert not np.may_share_memory(
                named_arr[j, ...], named_arr.__getattr__(name)
            )


@pytest.mark.parametrize("NamedArray", [NamedCupyArray, NamedNumpyArray])
@pytest.mark.parametrize("copy", [True, False])
def test_copy(NamedArray: callable, copy: bool):
    """
    assert copy=True returns a copy while copy=False returns a view
    args:
        NamedArray (callable) : NamedNumpyArray or NamedCupyArray
        copy (bool) : whether to return a copy (True) or a view (False)
    """
    x = np.linspace(1, 20, 20).reshape(2, 10)
    named_arr = NamedArray(x, ["a", "b"], copy=copy)
    x[0] = 99
    if copy:
        assert 99 not in named_arr
    else:
        assert np.all(named_arr.a == 99)


def test_namedcupy_to_namednumpy():
    """
    assert NamedCupyArray.asnamednumpy returns NamedNumpyArray
    """
    x = np.linspace(1, 50, 50).reshape(2, 5, 5)
    x_ncp = NamedCupyArray(x, ["a", "b"])
    x_nnp = x_ncp.asnamednumpy()
    assert x_nnp.xp == "numpy"


@pytest.mark.parametrize("NamedArray", [NamedCupyArray, NamedNumpyArray])
def test_save_load(NamedArray: callable, data_directory: str = "tests/data/"):
    """
    write NamedArray as a .npy file and then load it
    args:
        NamedArray (callable) : NamedNumpyArray or NamedCupyArray
        data_directory (str) : path to data folder
    """
    x = np.linspace(1, 50, 50).reshape(2, 5, 5)
    x_nnp = NamedArray(x, ["a", "b"])

    # save data
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    np.save(data_directory + "saveloadtest.npy", x_nnp)

    # load data
    loaded_x_nnp = np.load(data_directory + "saveloadtest.npy")
    assert np.all(x_nnp == loaded_x_nnp)


@pytest.mark.parametrize("NamedArray", [NamedCupyArray, NamedNumpyArray])
def test_redundant_init(NamedArray: callable):
    """
    create an instance from an instance
    args:
        NamedArray (callable) : NamedNumpyArray or NamedCupyArray
    """
    x = np.linspace(1, 50, 50).reshape(2, 5, 5)
    x_nnp = NamedArray(x, ["a", "b"])
    x_nnp_nnp = NamedArray(x_nnp, ["a", "b"])

    assert x_nnp.variable_indices == x_nnp_nnp.variable_indices
    assert x_nnp.variable_names == x_nnp_nnp.variable_names
    assert x_nnp.variable_name_set == x_nnp_nnp.variable_name_set
    assert x_nnp.xp == x_nnp_nnp.xp
    assert np.all(x_nnp == x_nnp_nnp)


@pytest.mark.parametrize("NamedArray", [NamedCupyArray, NamedNumpyArray])
@pytest.mark.parametrize("rename", [{"a": "A", "e": "E"}, ["A", "b", "c", "d", "E"]])
def test_rename_variables(NamedArray: callable, rename: Union[dict, Iterable]):
    """
    test variable renaming
    args:
        NamedArray (callable) : NamedNumpyArray or NamedCupyArray
        new_names (dict or Iterable) : map from old names to new names, or series of new names
    """
    x = np.linspace(1, 125, 125).reshape(5, 5, 5)
    x_nnp1 = NamedArray(x, ["a", "b", "c", "d", "e"])
    x_nnp2 = x_nnp1.copy()
    x_nnp2.rename_variables(rename)
    assert np.all(x_nnp1.a == x_nnp2.A)
    assert np.all(x_nnp1.b == x_nnp2.b)
    assert np.all(x_nnp1.c == x_nnp2.c)
    assert np.all(x_nnp1.d == x_nnp2.d)
    assert np.all(x_nnp1.e == x_nnp2.E)


@pytest.mark.parametrize("NamedArray", [NamedCupyArray, NamedNumpyArray])
@pytest.mark.parametrize("name", [[], "cat", ["fish", "monkey"]])
def test_remove(NamedArray: callable, name: Union[str, Iterable]):
    """
    test deleting variables
    args:
        NamedArray (callable) : NamedNumpyArray or NamedCupyArray
        name (str or Iterable) : names of variables to remove
    """
    x = np.linspace(1, 125, 125).reshape(5, 5, 5)
    x_nnp = NamedArray(x, ["cat", "dog", "fish", "monkey", "snake"])
    shorter_x_nnp = x_nnp.remove(name)

    remove_length = 1 if isinstance(name, str) else len(name)
    assert shorter_x_nnp.shape[0] + remove_length == x_nnp.shape[0]
    assert np.all(x_nnp.dog == shorter_x_nnp.dog)
    assert np.all(x_nnp.snake == shorter_x_nnp.snake)


@pytest.mark.parametrize("NamedArray", [NamedCupyArray, NamedNumpyArray])
def test_merge(NamedArray: callable):
    """
    test merging two instances of NamedArray
    args:
        NamedArray (callable) : NamedNumpyArray or NamedCupyArray
    """
    x1 = np.linspace(1, 125, 125).reshape(5, 5, 5)
    x_nnp1 = NamedArray(x1, ["rat", "squirrel", "chipmunk", "ferret", "beaver"])
    x2 = np.linspace(126, 250, 125).reshape(5, 5, 5)
    x_nnp2 = NamedArray(x2, ["rat", "dolphin", "chipmunk", "shark", "beaver"])
    merged = x_nnp1.merge(x_nnp2)
    assert np.all(merged.squirrel == x_nnp1.squirrel)
    assert np.all(merged.ferret == x_nnp1.ferret)
    assert np.all(merged.rat == x_nnp1.rat)
    assert np.all(merged.shark == x_nnp2.shark)
