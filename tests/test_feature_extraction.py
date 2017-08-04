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


def test_get_descriptor_vector(test_hists):
    descriptor = fd.FeatureDescriptor(test_hists)
    vector = descriptor.get_vector()
    total_length = 0
    for hist in descriptor.hists:
        total_length += len(hist)
    assert len(vector) == total_length


def test_get_reduced_descriptor(test_hists, test_image):
    hsv = test_hists
    h = hsv[0]
    s = hsv[1]
    v = hsv[2]
    d_h = fd.FeatureDescriptor(test_image, s=False, v=False)
    assert len(d_h.get_vector()) == len(h)
    d_hs = fd.FeatureDescriptor(test_image, v=False)
    assert len(d_hs.get_vector()) == len(h) + len(s)
    d_hsv = fd.FeatureDescriptor(test_image)
    assert len(d_hsv.get_vector()) == len(h) + len(s) + len(v)
    d_s = fd.FeatureDescriptor(test_image, h=False, v=False)
    assert len(d_s.get_vector()) == len(s)
    d_sv = fd.FeatureDescriptor(test_image, h=False)
    assert len(d_sv.get_vector()) == len(s) + len(v)
    d_v = fd.FeatureDescriptor(test_image, h=False, s=False)
    assert len(d_v.get_vector()) == len(v)


def test_get_frame_as_xml(test_hists):
    timestamp = lt.Timestamp.from_str("00:41:36.96")
    frame_1 = fd.VFrame(timestamp, fd.FeatureDescriptor(test_hists))
    frame_1_xml = fx.get_vframe_as_xml(frame_1)
    frame_2 = fd.VFrame(timestamp, fd.FeatureDescriptor([[], [], []]))
    frame_2_xml = fx.get_vframe_as_xml(frame_2)
    assert not hlp.xml_elements_equal(frame_1_xml, frame_2_xml)


@pytest.fixture()
def test_get_video_as_xml(test_hists):
    hsv = test_hists
    h = hsv[0]
    s = hsv[1]
    v = hsv[2]

    vd_h = fx.get_xml_from_videofile(cft.exp_video, s=False, v=False)
    assert len(vd_h[0][0]) == 1
    assert len(vd_h[0][0][0].text.split(";")) == len(h)
    vd_hs = fx.get_xml_from_videofile(cft.exp_video, v=False)
    assert len(vd_hs[0][0]) == 2
    assert len(vd_hs[0][0][0].text.split(";")) == len(h)
    assert len(vd_hs[0][0][1].text.split(";")) == len(s)
    vd_hsv = fx.get_xml_from_videofile(cft.exp_video)
    assert len(vd_hsv[0][0]) == 3
    assert len(vd_hsv[0][0][0].text.split(";")) == len(h)
    assert len(vd_hsv[0][0][1].text.split(";")) == len(s)
    assert len(vd_hsv[0][0][2].text.split(";")) == len(v)
    vd_s = fx.get_xml_from_videofile(cft.exp_video, h=False, v=False)
    assert len(vd_s[0][0]) == 1
    assert len(vd_s[0][0][0].text.split(";")) == len(s)
    vd_sv = fx.get_xml_from_videofile(cft.exp_video, h=False)
    assert len(vd_sv[0][0]) == 2
    assert len(vd_sv[0][0][0].text.split(";")) == len(s)
    assert len(vd_sv[0][0][1].text.split(";")) == len(v)
    vd_v = fx.get_xml_from_videofile(cft.exp_video, h=False, s=False)
    assert len(vd_v[0][0]) == 1
    assert len(vd_v[0][0][0].text.split(";")) == len(v)

    assert len(vd_hsv)
    return vd_hsv


def test_write_video_xml(test_get_video_as_xml):
    fx.write_video_to_xml(test_get_video_as_xml, cft.exp_video_ft)
    assert hlp.file_length(cft.exp_video_ft)


def test_read_video_frames():
    video = fx.get_video_from_xml(cft.exp_video_ft)
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


def test_get_ft_vec_list():
    video = fd.Video(xmlpath=cft.training_video_ft)
    ft_vec_list = video.get_featurevector_list()
    assert len(ft_vec_list)
