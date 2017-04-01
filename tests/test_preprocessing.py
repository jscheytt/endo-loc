from .context import sample

import pytest

import prep.preprocessor as pre
import helper.helper as hlp

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


def test_normalize_lists(feature):
    max_val = 10
    ft_vec_list_norm = pre.normalize_ft_vec_list(feature * 3, max_val)
    maxval = hlp.maxval_of_2dlist(ft_vec_list_norm)
    assert 0 <= maxval <= max_val
