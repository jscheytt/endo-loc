import os

import pytest

import feature_extraction.ft_extractor as fx

res_dir = "res" + os.sep

example_img = res_dir + "test_image.jpg"  # alt: "test_image_2.jpg"
example_vid = res_dir + "test_video_1s.mp4"  # alt: "test_video_1s.mp4", "test_video_100s.avi"
test_video_xml_filename = res_dir + "test_video_features.xml"
test_ilabel = res_dir + "test_ilabel.ass"
training_video = res_dir + "video_train.mp4"
training_labels = res_dir + "video_train.ass"


@pytest.fixture(scope="module")
def test_image():
    return fx.load_image(example_img)
