import numpy as np
import sklearn.preprocessing as skpre


def get_array(feature):
    """
    Return a numpy.ndarray suited for SVM learning.
    :param feature: 1-dimensional list
    :return: np.array(feature, dtype=float64, order= 'C')
    """
    return np.array(feature, dtype=np.float64, order='C')


def normalize_ft_vec_list(ft_vec_list, max_val):
    """
    Normalize list to values in [0, max_val] as ndarray.
    :param ft_vec_list: 1-dimensional list
    :param max_val: Maximum value of the array
    :return: numpy.ndarray with values [0, max_val]
    """
    ft_vec_list_norm_np = []
    max_val_fl = float(max_val)
    for ft_vec in ft_vec_list:
        ft_vec_norm = [val / max_val_fl for val in ft_vec]
        ft_vec_np = get_array(ft_vec_norm)
        ft_vec_list_norm_np.append(ft_vec_np)

    return ft_vec_list_norm_np
