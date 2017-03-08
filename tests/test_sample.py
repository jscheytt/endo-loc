from .context import sample

import tests.conftest as cft
import sample.sample as s


def test_get_svclassifier():
    svc = s.get_svclassifier(cft.training_video_ft, cft.training_labels)
    assert svc is not None
    assert svc.n_support_[0] > 0
