from .context import sample

import tests.conftest as cft
import sample.sample as s


def test_get_svclassifier():
    data_train, targets_train = s.get_data_and_targets(cft.training_video_ft, cft.training_labels)
    svc = s.get_svclassifier(data_train, targets_train)
    assert svc is not None
    assert svc.n_support_[0] > 0


def test_get_evaluation():
    data_train, targets_train = s.get_data_and_targets(cft.training_video_ft, cft.training_labels)
    svc = s.get_svclassifier(data_train, targets_train)
    data_eval, targets_eval = s.get_data_and_targets(cft.eval_video_ft, cft.eval_labels)
    evaluation = s.get_evaluation(svc, data_eval, targets_eval)
    assert evaluation != ""
