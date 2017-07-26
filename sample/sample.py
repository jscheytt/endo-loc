from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sk_mt
import sklearn.model_selection as sk_ms
from sklearn.externals import joblib
from sklearn.svm import SVC

from debug.debug import LogCont
import helper.helper as hlp
import prep.preprocessor as pre


def get_svclassifier(X=None, y=None, C: float = 1.0, gamma: Union[float, str] = 'auto'):
    """
    Fit a SVM classifier to the given training data.
    :param X: List of feature vectors
    :param y: List of labels
    :param C: 
    :param gamma: 
    :return: The learned classifier
    """
    clf = SVC(C=C, gamma=gamma, class_weight="balanced")
    if X is not None and y is not None:
        with LogCont("Fit SVM to data"):
            clf.fit(X, y)
    return clf


def get_evaluation_report(classifier, X, y, predicted=None):
    """
    Get an evaluation report on the prediction correctness of the classifier.
    :param classifier:
    :param X:
    :param y:
    :param predicted: predicted labels on data
    :return: evaluation report (str), confusion matrix (array)
    """
    expected = y
    if predicted is None:
        predicted = get_prediction(X, classifier)
    eval_report = sk_mt.classification_report(expected, predicted)
    conf_mat = get_confusion_mat(expected, predicted)
    return eval_report, conf_mat


def get_prediction(X, classifier):
    """
    Make classifier predict labels on data.
    :param X:
    :param classifier:
    :return: array of predicted labels
    """
    with LogCont("Predict on test data"):
        predicted = classifier.predict(X)
    return predicted


def get_grid_search(X, y):
    """
    Perform Grid Search on dataset with SVM classifier.
    :param X: 
    :param y: 
    :return: the C_range and gamma_range of the search, then the GridSearch obj itself
    """
    C_range = np.logspace(-1, 5, 7)
    gamma_range = np.logspace(-6, 2, 9)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = sk_ms.StratifiedShuffleSplit(n_splits=5, random_state=0)
    grid = sk_ms.GridSearchCV(SVC(class_weight="balanced"), param_grid,
                              scoring='f1', n_jobs=-1, cv=cv, error_score=0)
    with LogCont("Perform grid search"):
        grid.fit(X, y)
        # TODO externalize this to some main method
        hlp.log("The best parameters are %s with a score of %0.2f"
                % (grid.best_params_, grid.best_score_))

    return C_range, gamma_range, grid


def get_best_params(X, y):
    """
    Perform a grid search and return the best parameters.
    :param X: 
    :param y: 
    :return: 
    """
    _, _, grid_search = get_grid_search(X, y)
    return grid_search.best_params_


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


def predict_single_ft_vec(clf, ft_vec):
    """
    Predict the class label on a single feature vector.
    :param clf: classifier to be used for prediction
    :param ft_vec: feature vector without label
    :return: class label value
    """
    value = clf.predict(ft_vec)
    return value


def calc_scores_on_both_classes(y, y_pred, n_folds, metric):
    """
    Evaluate a score defined by metric for both classes on every partition.
    :param y:
    :param y_pred: predicted labels
    :param n_folds: number of folds
    :param metric: metric to be assessed over the predictions
    :return: scores of both classes, array of floats with shape (2, n_folds)
    """
    y_true = np.array(y)
    y_true_inv = 1 - y_true
    y_pred = np.array(y_pred)
    y_pred_inv = 1 - y_pred

    folds_y_true = np.array_split(y_true, n_folds)
    folds_y_pred = np.array_split(y_pred, n_folds)
    folds_y_true_inv = np.array_split(y_true_inv, n_folds)
    folds_y_pred_inv = np.array_split(y_pred_inv, n_folds)

    def scorer(folds):
        return metric(folds[0], folds[1])

    scores_class0 = list(map(scorer, zip(folds_y_true, folds_y_pred)))
    scores_class1 = list(map(scorer, zip(folds_y_true_inv, folds_y_pred_inv)))

    return [scores_class0, scores_class1]


def get_crossval_evaluation(X, y, n_folds=10, print_scores=False, clf=None, files_as_folds=False,
                            do_subsampling=True):
    """
    Perform cross validation and get evaluation report.
    :param X:
    :param y:
    :param n_folds: number of folds
    :param print_scores: print all scores to stdout
    :param clf: classifier object for learning
    :param files_as_folds: get partitions from files
    :param do_subsampling: perform subsampling in case of files as folds
    :return: str with scores mean and std. deviation,
    and the predicted labels
    """
    if clf is None:
        clf = get_svclassifier(X, y)

    with LogCont("Calculate cross validation"):
        if not files_as_folds and n_folds is not None:
            fold = sk_ms.StratifiedKFold(n_folds)
            y_pred = sk_ms.cross_val_predict(clf, X, y, cv=fold)
        else:
            y_pred = crossval_predict_files_folds(clf, X, y, do_subsampling=do_subsampling)
            y = np.concatenate(y)

    scores = calc_scores_on_both_classes(y, y_pred, n_folds,
                                         lambda x1, x2: sk_mt.f1_score(x1, x2, average="binary"))
    if print_scores:
        hlp.log(scores)

    report = ""
    for idx, score in enumerate(scores):
        pattern = "F1 score class " + str(idx) + ": %0.4f (+/- %0.4f)\n"
        report += pattern % (np.mean(score), np.std(score))

    return report, y_pred


def crossval_predict_files_folds(clf, X_list, y_list, do_subsampling=True):
    """
    Perform cross validation on lists of data and targets.
    Use the files / list entries as folds.
    :param clf: classifier object
    :param X_list: list of data arrays
    :param y_list: list of target arrays
    :param do_subsampling: perform subsampling on selection (not on eval data)
    :return:
    """
    y_pred = []
    for idx, X_eval in hlp.reverse_enum(X_list):
        # Create selection
        X_sel = X_list[:idx] + X_list[idx + 1:]
        y_sel = y_list[:idx] + y_list[idx + 1:]
        # Concatenate
        X_sel_comb = np.concatenate(X_sel)
        y_sel_comb = np.concatenate(y_sel)
        if do_subsampling:  # Perform subsampling
            pre.balance_class_sizes(X_sel_comb, y_sel_comb)

        clf.fit(X_sel_comb, y_sel_comb)
        y_pred_eval = get_prediction(X_eval, clf)
        y_pred += list(y_pred_eval)
    return y_pred


def get_confusion_mat(y_train, y_eval):
    """
    Calculate the confusion matrix.
    :param y_train: 
    :param y_eval: 
    :return: array of shape [n_classes, n_classes]
    """
    return sk_mt.confusion_matrix(y_train, y_eval)
