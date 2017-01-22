from .context import sample
import pytest

import feature_extraction.feature_descriptor as fd
import label_import.helper as hlp
import label_import.label_importer as li
import tests.conftest as cft


@pytest.mark.skip(reason="contained in test_get_labels_from_file")
def test_read_file():
    file_cont = li.LabelImporter.get_textfile_as_str(cft.test_ilabel)
    assert len(file_cont)


def test_get_label_obj():
    line = "Dialogue: 0,0:32:08.42,0:32:09.26,Default,,0,0,0,,MOVING_OUT"
    label_obj = li.LabelImporter.get_label_from_line(line)
    assert len(label_obj) and isinstance(label_obj, li.ILabel)


def test_get_labels_from_file():
    ilabels = li.LabelImporter.get_labels_from_file(cft.test_ilabel)
    assert len(ilabels)


def test_comp_timestamps():
    t1 = hlp.Timestamp("0:34:25.20")
    t2 = hlp.Timestamp("0:34:27.12")
    assert t1 < t2
    assert t2 > t1
    assert t1 == t1
    assert t1 <= t2
    assert t2 >= t1


def test_get_label_from_timestamp():
    t1 = hlp.Timestamp("0:00:00.00")
    t2 = hlp.Timestamp("0:34:25.20")
    val1 = li.ILabelValue.OUT
    val2 = li.ILabelValue.MOVING_IN
    labels = li.LabelImporter.get_labels_from_file(cft.test_ilabel)
    video = fd.Video(labels=labels)
    assert val1 == video.get_label_from_timestamp(t1).value
    assert val2 == video.get_label_from_timestamp(t2).value