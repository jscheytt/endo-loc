import pytest

from .context import sample

import tests.conftest as cft
import sample.sample as s
import prep.preprocessor as pre


def test_get_svclassifier():
    X_train, y_train = pre.get_data_and_targets(cft.training_video_ft, cft.training_label_list)
    svc = s.get_svclassifier(X_train, y_train)
    assert svc is not None
    assert svc.n_support_[0] > 0


def test_get_evaluation():
    X_train, y_train = pre.get_data_and_targets(cft.training_video_ft, cft.training_label_list)
    svc = s.get_svclassifier(X_train, y_train)
    X_eval, y_eval = pre.get_data_and_targets(cft.eval_video_ft, cft.eval_label_list)
    evaluation = s.get_evaluation(svc, X_eval, y_eval)
    assert evaluation != ""


def test_get_grid_search():
    X, y = pre.get_data_and_targets(cft.training_video_ft, cft.training_label_list, do_subsampling=True)
    C_range, gamma_range, grid = s.get_grid_search(X, y)
    assert grid is not None


@pytest.mark.skip(reason="GUI")
def test_plot_grid_search_results():
    X, y = pre.get_data_and_targets(cft.training_video_ft, cft.training_label_list, do_subsampling=True)
    C_range, gamma_range, grid = s.get_grid_search(X, y)
    s.plot_grid_search_results(grid, C_range, gamma_range)
