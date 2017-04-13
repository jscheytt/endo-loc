from .context import sample

import pytest

import prep.preprocessor as pre
import helper.helper as hlp
import tests.conftest as cft

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


def test_balance_class_sizes():
    X, y = pre.get_data_and_targets(cft.training_video_ft, cft.training_label_list)
    # X_bal, y_bal = pre.balance_class_sizes(X, y)
    pre.balance_class_sizes(X, y)
    # classes = pre.get_indices_of_classes(X_bal, y_bal)
    classes = pre.get_indices_of_classes(X, y)
    for c in classes:
        assert pre.classes_balanced(classes[0], c)


def test_classes_balanced():
    classes = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5],
    ]
    c_imbal = [1, 2, 3, 4]
    for c in classes:
        assert pre.classes_balanced(classes[0], c)
        assert pre.classes_balanced(c, classes[0])
    assert not pre.classes_balanced(classes[0], c_imbal)
    assert not pre.classes_balanced(c_imbal, classes[0])


def test_get_indices_of_classes():
    X = [[1, 2], [1, 2], [2, 3], [2, 3], [3, 4], [4, 5]]
    y = [0, 1, 1, 0, 0, 1]
    classes = pre.get_indices_of_classes(X, y)
    assert len(classes[0]) == len(classes[1])
