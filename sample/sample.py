from sklearn import svm, metrics

import feature_extraction.ft_descriptor as fd
import label_import.label_importer as li
import preprocessing.preprocessor as pre
from debug.debug import LogCont


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


def get_svclassifier(data, targets):
    """
    Fit a SVM classifier to the given training data.
    :param data: List of feature vectors
    :param targets: List of labels
    :return: The learned classifier
    """
    clf = svm.SVC(class_weight="balanced")
    with LogCont("Fit SVM to data"):
        clf.fit(data, targets)
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
    with LogCont("Predict on test data"):
        predicted = classifier.predict(data)
    return metrics.classification_report(expected, predicted)
