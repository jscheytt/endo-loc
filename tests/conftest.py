import os

import pytest

from sample.debug import Debug

res_dir = "res" + os.sep

example_img = res_dir + "test_image.jpg"  # alt: "test_image_2.jpg"
example_vid = res_dir + "test_video_1s.mp4"  # alt: "test_video_1s.mp4", "test_video_100s.avi"
test_video_xml_filename = res_dir + "test_video_features.xml"
test_ilabel = res_dir + "test_ilabel.ass"


@pytest.fixture(scope="module")
def test_image():
    return Debug.load_image(example_img)
