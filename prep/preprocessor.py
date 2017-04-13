import math

import numpy as np
from sklearn.model_selection import train_test_split

from debug.debug import LogCont
from feature_extraction import ft_descriptor as fd
from label_import import label_importer as li
from helper.helper import reverse_enum


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


def get_data_and_targets(xmlpath, labellistpath, do_subsampling=False):
    """
    Get sklearn-conform lists of features (data) and labels (targets).
    :param xmlpath: Path to the feature XML file
    :param labellistpath: Path to the label list CSV file
    :param do_subsampling: Subsample majority class
    :return:
    """
    with LogCont("Import label list"):
        label_list = li.read_label_list(labellistpath)
    with LogCont("Import feature list"):
        video = fd.Video(xmlpath=xmlpath, label_list=label_list)
        ft_vec_list = video.get_featurevector_list()
    with LogCont("Preprocess feature list"):
        norm_ft_vec_list = normalize_ft_vec_list(ft_vec_list, max_val=video.numpx)
    if do_subsampling:
        balance_class_sizes(norm_ft_vec_list, label_list)
    return norm_ft_vec_list, label_list


def get_train_test_data_targets(X, y):
    """
    Split 1 feature vector and corresponding label list into training and test data. 
    :param X: 
    :param y: 
    :return: 
    """
    with LogCont("Split data/targets into training and test set"):
        return train_test_split(X, y)


def classes_balanced(c1, c2):
    """
    Check if the ratio of class sizes is within the accepted range for balance.
    :param c1: 
    :param c2: 
    :return: 
    """
    return 0.5 <= len(c1) / len(c2) <= 2


def get_indices_of_classes(X, y):
    """
    Get indices of all feature vectors of all classes, grouped by their class label. 
    :param X: list of feature vectors
    :param y: class labels, integers from 0 to n
    :return: 2D list of feature vector indices grouped by class label
    """
    num_classes = len(set(y))
    classes = []
    for i in range(num_classes):
        classes.append([])
    for idx, elem in enumerate(X):
        class_idx = y[idx]
        classes[class_idx].append(idx)
    return classes


def balance_class_sizes(X, y):
    """
    Balance classes for binary classification.
    :param X: list of feature vectors
    :param y: class labels (0 and 1)
    :return: 2 balanced classes
    """
    with LogCont("Subsample majority class"):
        cl0, cl1 = get_indices_of_classes(X, y)
        if len(cl0) > len(cl1):
            len_maj = len(cl0)
            len_min = len(cl1)
            cl_maj = cl0
        else:
            len_maj = len(cl1)
            len_min = len(cl0)
            cl_maj = cl1
        n = math.floor(len_maj / len_min)
        for idx, elem in reverse_enum(cl_maj):
            if idx % n != 0:
                del X[elem]
                del y[elem]
