from .context import sample

import pytest

from debug.debug import Debug as Dbg
import feature_extraction.ft_extractor as fx
import tests.conftest as cft


def test_load_image(test_image):
    assert len(test_image)


@pytest.mark.skip(reason="GUI")
def test_display_image(test_image):
    Dbg.display_image(test_image)


def test_get_histograms_rgb(test_image):
    hists = Dbg.get_histograms_rgb(test_image)
    for hist in hists:
        assert len(hist)


@pytest.mark.skip(reason="GUI")
def test_plot_histogram_rgb(test_image):
    hists = Dbg.get_histograms_rgb(test_image)
    Dbg.plot_histograms(hists)


@pytest.mark.skip(reason="GUI")
def test_plot_histogram_hsv(test_image):
    hists = fx.get_histograms_hsv(test_image)
    Dbg.plot_histograms(hists)


@pytest.mark.skip(reason="GUI")
def test_display_video_histogram_hsv():
    Dbg.plot_histograms_live(cft.example_vid)
    # TODO Fail if not displayed
