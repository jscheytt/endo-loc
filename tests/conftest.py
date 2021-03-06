import os

import pytest

import feature_extraction.ft_extractor as fx
import prep.preprocessor as pre
import helper.helper as hlp

res_dir = os.path.join("tests", "res") + os.sep

test_img = res_dir + "test_image.jpg"  # alt: "test_image_2.jpg"
test_img_rot = res_dir + "test_image_rot.jpg"
exp_video = res_dir + "video_test_short.mp4"
exp_video_ft = res_dir + "video_test_short_hsv.xml"
test_ilabel = res_dir + "test_ilabel.ass"
test_label_list = res_dir + "test_label_list.csv"
data_targets_directory = hlp.get_full_path_from_projroot(os.path.join("tests", "res", "data_targets_test"))

training_video = res_dir + "video_train.mp4"
training_video_ft = res_dir + "video_train_hsv.xml"
training_labels = res_dir + "video_train.ass"
training_label_list = res_dir + "video_train.csv"

eval_video = res_dir + "video_eval.mp4"
eval_video_ft = res_dir + "video_eval_hsv.xml"
eval_labels = res_dir + "video_eval.ass"
eval_label_list = res_dir + "video_eval.csv"

clf_dump = res_dir + "svclassifier.pkl"
list_export = res_dir + "list.txt"


@pytest.fixture(scope="module")
def test_image():
    img = fx.load_image(test_img)
    assert len(img)
    return img


@pytest.fixture(scope="module")
def test_image_rot():
    img = fx.load_image(test_img_rot)
    assert len(img)
    return img


@pytest.fixture(scope='module')
def test_data_and_targets():
    X, y = pre.get_data_and_targets(training_video_ft, training_label_list)
    assert len(X)
    assert len(y)
    return X, y


@pytest.fixture(scope='module')
def test_data_and_targets_subsampled():
    X, y = pre.get_data_and_targets(training_video_ft, training_label_list, do_subsampling=True)
    assert len(X)
    assert len(y)
    return X, y


@pytest.fixture(scope='module')
def eval_data_and_targets():
    X, y = pre.get_data_and_targets(eval_video_ft, eval_label_list)
    assert len(X)
    assert len(y)
    return X, y


@pytest.fixture(scope='module')
def test_get_multiple_data_and_targets():
    X_list, y_list = pre.get_multiple_data_and_targets(data_targets_directory, do_subsampling=True, do_concat=False)
    assert len(X_list) > 1
    assert len(y_list) > 1
    assert len(X_list) == len(y_list)
    return X_list, y_list
