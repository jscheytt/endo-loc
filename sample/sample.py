from sklearn import svm

import feature_extraction.ft_descriptor as fd
import label_import.label_importer as li
import preprocessing.preprocessor as pre


def get_svclassifier(xmlpath, labelspath):
    labels = li.get_labels_from_file(labelspath)
    video = fd.Video(xmlpath=xmlpath, labels=labels)
    label_list = video.get_label_list()

    ft_vec_list = video.get_featurevector_list()
    norm_ft_vec_list = pre.normalize_ft_vec_list(ft_vec_list, max_val=video.numpx)

    clf = svm.SVC()
    clf.fit(norm_ft_vec_list, label_list)

    return clf
