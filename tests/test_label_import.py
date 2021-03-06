from .context import sample

import pytest

import feature_extraction.ft_descriptor as fd
import label_import.label as lb
import label_import.label_importer as li
import label_import.timestamp as lt
import tests.conftest as cft
import helper.helper as hlp


@pytest.mark.skip(reason="contained in test_get_labels_from_file")
def test_read_file():
    file_cont = li.get_textfile_as_str(cft.test_ilabel)
    assert len(file_cont)


def test_get_label_obj():
    line = "Dialogue: 0,0:32:08.42,0:32:09.26,Default,,0,0,0,,MOVING_OUT"
    label_obj = li.get_label_from_line(line)
    assert len(label_obj) and isinstance(label_obj, lb.ILabel)


def test_get_labels_from_file():
    ilabels = li.read_labels(cft.test_ilabel)
    assert len(ilabels)


def test_comp_timestamps():
    t1 = lt.Timestamp.from_str("0:34:25.20")
    t2 = lt.Timestamp.from_str("0:34:27.12")
    assert t1 < t2
    assert t2 > t1
    assert t1 == t1
    assert t1 <= t2
    assert t2 >= t1


def test_get_label_from_timestamp():
    t1 = lt.Timestamp.from_str("0:00:00.00")
    t2 = lt.Timestamp.from_str("0:01:19.76")
    val1 = lb.ILabelValue.OUT
    val2 = lb.ILabelValue.IN
    labels = li.read_labels(cft.test_ilabel)
    video = fd.Video(labels=labels)
    assert val1 == video.get_label(t1).value
    assert val2 == video.get_label(t2).value


def test_get_timestamp():
    frame_number = 62424
    timestamp_str = "00:41:36.96"
    fps = 25
    timestamp_obj = lt.Timestamp.from_frameidx_fps(frame_number, fps)
    assert timestamp_str == timestamp_obj.to_str()


def test_get_frame_number():
    frame_number = 62424
    timestamp_str = "00:41:36.96"
    fps = 25
    timestamp_obj = lt.Timestamp.from_str(timestamp_str)
    assert frame_number == timestamp_obj.get_frameidx(fps)


def test_reduce_label_value():
    for l in lb.ILabelValue:
        red_l = li.reduce_label_value(l)
        assert red_l is not lb.ILabelValue.MOVING_IN
        assert red_l is not lb.ILabelValue.MOVING_OUT
        assert red_l is not lb.ILabelValue.IN_BETWEEN
        assert red_l is not lb.ILabelValue.EXIT


def test_fill_label_list():
    labels = li.read_labels(cft.training_labels)
    video = fd.Video(xmlpath=cft.training_video_ft, labels=labels)
    video.fill_label_list()
    label_list = video.label_list
    assert len(label_list)
    assert len(video.frames) == len(label_list)


def test_write_label_list():
    labels = li.read_labels(cft.training_labels)
    video = fd.Video(xmlpath=cft.training_video_ft, labels=labels)
    video.write_label_list(cft.test_label_list)
    assert hlp.file_length(cft.test_label_list)


def test_read_label_list():
    label_list = li.read_label_list(cft.test_label_list)
    assert len(label_list)
