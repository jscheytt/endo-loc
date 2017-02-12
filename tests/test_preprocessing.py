from .context import sample

import pytest

import preprocessing.preprocessor as pre
import numpy as np


@pytest.fixture()
def feature():
    hist = [1, 2, 5, 0]
    return [hist + hist + hist]


@pytest.fixture()
def ft_array(feature):
    return pre.get_array(feature)


def test_get_array(ft_array):
    assert isinstance(ft_array, np.ndarray)
    assert ft_array.dtype is np.dtype(np.float64)
    assert ft_array.flags['C']


def test_normalize_list(feature):
    max_val = 10
    ft_array_norm = pre.normalize_array(feature, max_val)
    assert 0 <= ft_array_norm.max() <= max_val
    assert ft_array_norm.min() == 0.0
