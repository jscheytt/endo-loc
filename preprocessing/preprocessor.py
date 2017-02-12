import numpy as np
import sklearn.preprocessing as skpre


def get_array(feature):
    """
    Return a numpy.ndarray suited for SVM learning.
    :param feature: 1-dimensional list
    :return: np.array(feature, dtype=float64, order= 'C')
    """
    return np.array(feature, dtype=np.float64, order='C')


def normalize_array(feature, max_val):
    """
    Normalize list to values in [0, max_val] as ndarray.
    :param feature: 1-dimensional list
    :param max_val: Maximum value of the array
    :return: numpy.ndarray with values [0, max_val]
    """
    ft_array = get_array(feature)
    scaler = skpre.MinMaxScaler(feature_range=(0, max_val))
    return scaler.fit_transform(ft_array)
