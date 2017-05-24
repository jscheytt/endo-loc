import pytest

from .context import sample

import tests.conftest as cft
import sample.sample as s
import helper.helper as hlp


@pytest.fixture(scope="module")
def test_get_svclassifier(test_data_and_targets):
    X_train, y_train = test_data_and_targets
    svc = s.get_svclassifier(X_train, y_train)
    assert svc is not None
    assert svc.n_support_[0] > 0
    return svc


def test_get_evaluation(test_get_svclassifier, eval_data_and_targets):
    svc = test_get_svclassifier
    X_eval, y_eval = eval_data_and_targets
    evaluation = s.get_evaluation_report(svc, X_eval, y_eval)
    assert len(evaluation)


@pytest.mark.skip(reason="Intensive")
def test_get_grid_search(test_data_and_targets_subsampled):
    X, y = test_data_and_targets_subsampled
    C_range, gamma_range, grid = s.get_grid_search(X, y)
    assert grid is not None


@pytest.mark.skip(reason="Intensive")
def test_get_best_params(test_data_and_targets_subsampled):
    X, y = test_data_and_targets_subsampled
    best_params = s.get_best_params(X, y)
    assert len(best_params)


@pytest.mark.skip(reason="GUI")
def test_plot_grid_search_results(test_data_and_targets_subsampled):
    X, y = test_data_and_targets_subsampled
    C_range, gamma_range, grid = s.get_grid_search(X, y)
    s.plot_grid_search_results(grid, C_range, gamma_range)


def test_export_classifier(test_get_svclassifier):
    svc = test_get_svclassifier
    s.write_classifier(svc, cft.clf_dump)
    assert hlp.file_length(cft.clf_dump)


def test_import_classifier():
    svc = s.read_classifier(cft.clf_dump)
    assert svc is not None
    assert svc.n_support_[0] > 0


@pytest.mark.skip(reason="Contained")
def test_get_crossval_scores(test_data_and_targets):
    X, y = test_data_and_targets
    n_folds = 2
    scores = s.get_crossval_scores(X, y, n_folds=n_folds)
    assert len(scores) == n_folds


def test_get_crossval_evaluation(test_data_and_targets):
    X, y = test_data_and_targets
    n_folds = 2
    evaluation = s.get_crossval_evaluation(X, y, n_folds=n_folds)
    assert len(evaluation) == 29  # fixed length of str
