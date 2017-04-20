import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

from debug.debug import LogCont


def get_svclassifier(X, y, C=1.0, gamma='auto'):
    """
    Fit a SVM classifier to the given training data.
    :param X: List of feature vectors
    :param y: List of labels
    :param C: 
    :param gamma: 
    :return: The learned classifier
    """
    clf = svm.SVC(C=C, gamma=gamma, class_weight="balanced")
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


def get_grid_search(X, y):
    """
    Perform Grid Search on dataset with SVM classifier.
    :param X: 
    :param y: 
    :return: the C_range and gamma_range of the search, then the GridSearch obj itself
    """
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, random_state=0)
    grid = GridSearchCV(SVC(class_weight="balanced"), param_grid,
                        scoring='f1', n_jobs=-1, cv=cv, error_score=0)
    with LogCont("Perform grid search"):
        grid.fit(X, y)
        # TODO externalize this to some main method
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

    return C_range, gamma_range, grid


def plot_grid_search_results(grid, C_range, gamma_range):
    """
    Plot heatmap resulting from Grid Search.
    Plotting taken from http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    :param grid: Grid Search obj
    :param C_range: 
    :param gamma_range: 
    :return: 
    """
    from matplotlib.colors import Normalize

    class MidpointNormalize(Normalize):
        """
        Utility function to move the midpoint of a colormap to be around
        the values of interest.
        """

        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))
    plt.figure(figsize=(8, 6))
    # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.get_cmap("hot"),
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()


def write_classifier(clf, filepath):
    """
    Write a classifier to the filepath.
    :param clf: classifier from sklearn
    :param filepath: 
    :return: 
    """
    joblib.dump(clf, filepath)


def read_classifier(filepath):
    """
    Read a classifier from filepath
    :param filepath: 
    :return: classifier 
    """
    return joblib.load(filepath)
