import pytest

from .context import sample

import tests.conftest as cft
import sample.sample as s
import vis.display as dsp
import vis.classify_live as cllv
import helper.helper as hlp
import label_import.label as ll
import vis.geometry as geom


@pytest.mark.skip(reason="GUI")
def test_display_video():
    dsp.display_video(cft.eval_video)


def test_get_screen_dims():
    screen_dims = dsp.get_screen_dims()
    assert len(screen_dims)


def test_resize_img(test_image):
    resized = geom.resize_img(test_image)
    assert hlp.imgs_different(test_image, resized)


def test_resize_for_fullscreen(test_image):
    screen_dim = dsp.get_screen_dims()
    resized = geom.resize_for_fullscreen(test_image)
    resized_dims = geom.get_img_dims(resized)
    assert resized_dims == screen_dim


# @pytest.mark.skip(reason="GUI")
def test_classify_live():
    cllv.clf = s.read_classifier(cft.clf_dump)
    dsp.process_video(cft.training_video, cllv.do_workflow)


@pytest.fixture(scope="module")
def test_live_ft_vec(test_image):
    ft_vec = cllv.get_live_ft_vec(test_image)
    assert ft_vec is not None
    assert ft_vec.shape != (1, 1)
    assert ft_vec.shape[0] > 2
    return ft_vec


@pytest.fixture(scope="module")
def test_ft_vec_label(test_live_ft_vec):
    clf = s.read_classifier(cft.clf_dump)
    label = cllv.predict_label(clf, test_live_ft_vec)
    assert label is not None
    assert isinstance(label, ll.ILabelValue)
    assert label is ll.ILabelValue.IN or label is ll.ILabelValue.OUT
    return label


def test_draw_label(test_image, test_ft_vec_label):
    img_copy = test_image.copy()
    cllv.draw_label(test_image, test_ft_vec_label)
    assert hlp.imgs_different(img_copy, test_image)
