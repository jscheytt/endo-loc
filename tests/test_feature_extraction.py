from .context import sample

import pytest

import feature_extraction.ft_descriptor as fd
import feature_extraction.ft_extractor as fx
import helper.helper as hlp
import label_import.timestamp as lt
import tests.conftest as cft


@pytest.fixture()
def test_hists(test_image):
    return fx.get_histograms_hsv(test_image)


def test_get_histograms_hsv(test_hists):
    hists_hsv = test_hists
    # hists_rgb = Debug.get_histograms_rgb(test_image)
    for hist_hsv in hists_hsv:
        assert len(hist_hsv)
        # TODO Fail if both histograms are equal


def test_get_descriptor_as_xml(test_hists):
    descriptor = fd.FeatureDescriptor(test_hists)
    xml = fx.get_descriptor_as_xml(descriptor)
    from lxml import etree
    assert len(xml) and isinstance(xml, etree._Element)


def test_get_frame_as_xml(test_hists):
    timestamp = lt.Timestamp.from_str("00:41:36.96")
    frame_1 = fd.VFrame(timestamp, fd.FeatureDescriptor(test_hists))
    frame_1_xml = fx.get_vframe_as_xml(frame_1)
    frame_2 = fd.VFrame(timestamp, fd.FeatureDescriptor([[], [], []]))
    frame_2_xml = fx.get_vframe_as_xml(frame_2)
    assert not hlp.xml_elements_equal(frame_1_xml, frame_2_xml)


@pytest.mark.skip(reason="Already contained in test_write_video_xml")
def test_get_video_as_xml():
    video_xml = fx.get_xml_from_videofile(cft.example_vid)
    assert len(video_xml)


def test_write_video_xml():
    video_xml = fx.get_xml_from_videofile(cft.example_vid)
    assert len(video_xml)
    fx.write_video_to_xml(video_xml, cft.test_video_xml_filename)
    assert hlp.file_length(cft.test_video_xml_filename)


def test_read_video_frames():
    video = fx.get_video_from_xml(cft.test_video_xml_filename)
    assert len(video.frames)


def test_get_img_numpx(test_image):
    img_width = 640
    img_height = 360
    img_numpx = img_width * img_height
    assert img_numpx == fx.get_img_numpx(test_image)


def test_video_add_frame():
    video = fd.Video()
    number_of_frames = len(video.frames)
    new_frame = fd.VFrame()
    video.add_frame(new_frame)
    new_number_of_frames = len(video.frames)
    assert number_of_frames != new_number_of_frames
