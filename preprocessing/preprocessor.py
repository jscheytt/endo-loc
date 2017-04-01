import numpy as np

from debug.debug import LogCont
from feature_extraction import ft_descriptor as fd
from label_import import label_importer as li


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


def get_data_and_targets(xmlpath, labellistpath):
    """
    Get sklearn-conform lists of features (data) and labels (targets).
    :param xmlpath: Path to the feature XML file
    :param labellistpath: Path to the label list CSV file
    :return:
    """
    with LogCont("Import label list"):
        label_list = li.read_label_list(labellistpath)
    with LogCont("Import feature list"):
        video = fd.Video(xmlpath=xmlpath, label_list=label_list)
        ft_vec_list = video.get_featurevector_list()
    with LogCont("Preprocess feature list"):
        norm_ft_vec_list = pre.normalize_ft_vec_list(ft_vec_list, max_val=video.numpx)
    return norm_ft_vec_list, label_list