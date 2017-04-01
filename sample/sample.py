from sklearn import svm, metrics

from debug.debug import LogCont


def get_svclassifier(X, y):
    """
    Fit a SVM classifier to the given training data.
    :param X: List of feature vectors
    :param y: List of labels
    :return: The learned classifier
    """
    clf = svm.SVC(class_weight="balanced")
    with LogCont("Fit SVM to data"):
        clf.fit(X, y)
    return clf


def get_evaluation(classifier, X, y):
    """
    Get an evaluation report on the prediction correctness of the classifier.
    :param classifier:
    :param X:
    :param y:
    :return: String containing the evaluation report
    """
    expected = y
    with LogCont("Predict on test data"):
        predicted = classifier.predict(X)
    return metrics.classification_report(expected, predicted)
