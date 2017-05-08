import os

import pytest

import feature_extraction.ft_extractor as fx
import prep.preprocessor as pre

res_dir = "tests" + os.sep + "res" + os.sep

test_img = res_dir + "test_image.jpg"  # alt: "test_image_2.jpg"
test_img_rot = res_dir + "test_image_rot.jpg"
example_vid = res_dir + "test_video_1s.mp4"  # alt: "test_video_1s.mp4", "test_video_100s.avi"
test_video_xml_filename = res_dir + "test_video_features.xml"
test_ilabel = res_dir + "test_ilabel.ass"
test_label_list = res_dir + "test_label_list.csv"
data_targets_directory = 'C:\\Users\\Josia\\Documents\\Dropbox\\Studium\\HuC\\CaMed ' \
                         'Masterprojekt\\endo-loc\\tests\\res\\data_targets_test'

training_video = res_dir + "video_train.mp4"
training_video_ft = res_dir + "video_train_ft.xml"
training_labels = res_dir + "video_train.ass"
training_label_list = res_dir + "video_train.csv"

eval_video = res_dir + "video_eval.mp4"
eval_video_ft = res_dir + "video_eval_ft.xml"
eval_labels = res_dir + "video_eval.ass"
eval_label_list = res_dir + "video_eval.csv"

clf_dump = res_dir + "svclassifier.pkl"


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
