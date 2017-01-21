from .context import sample

import pytest
from lxml import etree

import feature_extraction.feature_descriptor as fd
import feature_extraction.feature_extractor as fx
import feature_extraction.helper as hlp
import tests.conftest as cft


@pytest.fixture()
def test_hists(test_image):
    return fx.FeatureExtractor.get_histograms_hsv(test_image)


def test_get_histograms_hsv(test_hists):
    hists_hsv = test_hists
    # hists_rgb = Debug.get_histograms_rgb(test_image)
    for hist_hsv in hists_hsv:
        assert len(hist_hsv)
        # TODO Fail if both histograms are equal


def test_get_descriptor_as_xml(test_hists):
    descriptor = fd.FeatureDescriptor(test_hists)
    xml = fx.FeatureExtractor.get_descriptor_as_xml(descriptor)
    assert len(xml) and isinstance(xml, etree._Element)


def test_get_frame_as_xml(test_hists):
    # frame_number = 62424
    timestamp = "00:41:36.96"
    frame_1 = fd.VFrame(timestamp, fd.FeatureDescriptor(test_hists))
    frame_1_xml = fx.FeatureExtractor.get_vframe_as_xml(frame_1)
    frame_2 = fd.VFrame(timestamp, fd.FeatureDescriptor([[], [], []]))
    frame_2_xml = fx.FeatureExtractor.get_vframe_as_xml(frame_2)
    from tests.helper import xml_elements_equal
    assert not xml_elements_equal(frame_1_xml, frame_2_xml)


@pytest.mark.skip(reason="Already contained in test_write_video_xml")
def test_get_video_as_xml():
    video_xml = fx.FeatureExtractor.get_videofile_as_xml(cft.example_vid)
    assert len(video_xml)


def test_get_timestamp():
    frame_number = 62424
    timestamp = "00:41:36.96"
    fps = 25
    assert timestamp == hlp.get_timestamp(frame_number, fps)


def test_get_frame_number():
    frame_number = 62424
    timestamp = "00:41:36.96"
    fps = 25
    assert frame_number == hlp.get_frameidx(timestamp, fps)


def test_write_video_xml():
    video_xml = fx.FeatureExtractor.get_videofile_as_xml(cft.example_vid)
    fx.FeatureExtractor.write_video_to_xml(video_xml)


def test_read_video_frames():
    video = fx.FeatureExtractor.get_video_from_xml(cft.test_video_xml_filename)
    assert len(video.frames)
