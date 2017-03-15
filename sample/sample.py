from sklearn import svm, metrics

import feature_extraction.ft_descriptor as fd
import label_import.label_importer as li
import preprocessing.preprocessor as pre
import logging


def get_data_and_targets(xmlpath, labelspath):
    labels = li.read_labels(labelspath)
    video = fd.Video(xmlpath=xmlpath, labels=labels)
    logging.info("Expand labels list ...")
    label_list = video.get_label_list()
    logging.info("Labels list expanded.")

    ft_vec_list = video.get_featurevector_list()
    logging.info("Import feature list ...")
    norm_ft_vec_list = pre.normalize_ft_vec_list(ft_vec_list, max_val=video.numpx)
    logging.info("Feature list imported.")
    return norm_ft_vec_list, label_list


def get_svclassifier(data, targets):
    clf = svm.SVC()
    logging.info("Fit SVM to data ...")
    clf.fit(data, targets)
    logging.info("SVM fitted.")
    return clf


def get_evaluation(classifier, data, targets):
    expected = targets
    predicted = classifier.predict(data)
    return metrics.classification_report(expected, predicted)
