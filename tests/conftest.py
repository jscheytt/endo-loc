import os

import pytest

import feature_extraction.ft_extractor as fx

res_dir = "tests" + os.sep + "res" + os.sep

example_img = res_dir + "test_image.jpg"  # alt: "test_image_2.jpg"
example_vid = res_dir + "test_video_1s.mp4"  # alt: "test_video_1s.mp4", "test_video_100s.avi"
test_video_xml_filename = res_dir + "test_video_features.xml"
test_ilabel = res_dir + "test_ilabel.ass"
test_label_list = res_dir + "test_label_list.csv"

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
    return fx.load_image(example_img)
