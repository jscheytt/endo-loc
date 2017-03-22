from sklearn import svm, metrics

import feature_extraction.ft_descriptor as fd
import label_import.label_importer as li
import preprocessing.preprocessor as pre
import logging


def get_data_and_targets(xmlpath, labellistpath):
    """
    Get sklearn-conform lists of features (data) and labels (targets).
    :param xmlpath: Path to the feature XML file
    :param labellistpath: Path to the label list CSV file
    :return:
    """
    label_list = li.read_label_list(labellistpath)
    video = fd.Video(xmlpath=xmlpath, label_list=label_list)
    ft_vec_list = video.get_featurevector_list()
    logging.info("Import feature list ...")
    norm_ft_vec_list = pre.normalize_ft_vec_list(ft_vec_list, max_val=video.numpx)
    logging.info("Feature list imported.")
    return norm_ft_vec_list, label_list


def get_svclassifier(data, targets):
    """
    Fit a SVM classifier to the given training data.
    :param data: List of feature vectors
    :param targets: List of labels
    :return: The learned classifier
    """
    clf = svm.SVC()
    logging.info("Fit SVM to data ...")
    clf.fit(data, targets)
    logging.info("SVM fitted.")
    return clf


def get_evaluation(classifier, data, targets):
    """
    Get an evaluation report on the prediction correctness of the classifier.
    :param classifier:
    :param data:
    :param targets:
    :return: String containing the evaluation report
    """
    expected = targets
    predicted = classifier.predict(data)
    return metrics.classification_report(expected, predicted)
