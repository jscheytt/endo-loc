from .context import sample

import pytest

import debug.debug as dbg
import feature_extraction.ft_extractor as fx
import helper.helper as hlp
import tests.conftest as cft


@pytest.mark.skip(reason="GUI")
def test_display_image(test_image):
    dbg.display_image(test_image)


def test_get_histograms_rgb(test_image):
    hists = dbg.get_histograms_rgb(test_image)
    for hist in hists:
        assert len(hist)


@pytest.mark.skip(reason="GUI")
def test_plot_histogram_rgb(test_image):
    hists = dbg.get_histograms_rgb(test_image)
    dbg.plot_histograms(hists)


@pytest.mark.skip(reason="GUI")
def test_plot_histogram_hsv(test_image):
    hists = fx.get_histograms_hsv(test_image)
    dbg.plot_histograms(hists)


@pytest.mark.skip(reason="GUI")
def test_display_video_histogram_hsv():
    dbg.plot_histograms_live(cft.exp_video)
    # TODO Fail if not displayed


def test_log_surr():
    with dbg.LogCont("Do this action"):
        pass


def test_write_list_to_file():
    l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    dbg.write_list_to_file(l, cft.list_export)
    assert hlp.file_length(cft.list_export)
