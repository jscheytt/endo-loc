import pytest

from .context import sample

import tests.conftest as cft
import sample.sample as s
import prep.preprocessor as pre
import helper.helper as hlp


def test_get_svclassifier(test_data_and_targets):
    X_train, y_train = test_data_and_targets
    svc = s.get_svclassifier(X_train, y_train)
    assert svc is not None
    assert svc.n_support_[0] > 0


def test_get_evaluation(test_data_and_targets, eval_data_and_targets):
    X_train, y_train = test_data_and_targets
    svc = s.get_svclassifier(X_train, y_train)
    X_eval, y_eval = eval_data_and_targets
    evaluation = s.get_evaluation_report(svc, X_eval, y_eval)
    assert evaluation != ""


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


def test_export_classifier(test_data_and_targets):
    X, y = test_data_and_targets
    svc = s.get_svclassifier(X, y)
    s.write_classifier(svc, cft.clf_dump)
    assert hlp.file_length(cft.clf_dump)


def test_import_classifier():
    svc = s.read_classifier(cft.clf_dump)
    assert svc is not None
    assert svc.n_support_[0] > 0
