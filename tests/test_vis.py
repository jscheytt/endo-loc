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


def test_fill_for_fullscreen(test_image, test_image_rot):
    screen_width, screen_height = dsp.get_screen_dims()
    ratio_screen = screen_width / screen_height

    filled1 = geom.fill_img_for_fullscreen(test_image)
    filled1_width, filled1_height = geom.get_img_dims(filled1)
    ratio_filled1 = filled1_width / filled1_height

    filled2 = geom.fill_img_for_fullscreen(test_image_rot)
    filled2_width, filled2_height = geom.get_img_dims(filled2)
    ratio_filled2 = filled2_width / filled2_height

    assert ratio_filled1 == pytest.approx(ratio_screen, 1e-2)
    assert ratio_filled2 == pytest.approx(ratio_screen, 1e-2)


@pytest.mark.skip(reason="GUI")
def test_classify_live():
    cllv.CLF = s.read_classifier(cft.clf_dump)
    dsp.process_video(cft.training_video, cllv.display_predict_on_frame)


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


def test_draw_menu(test_image):
    img_copy = test_image.copy()
    cllv.draw_menu(test_image)
    assert hlp.imgs_different(img_copy, test_image)
