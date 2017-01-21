from .context import sample

import pytest

from feature_extraction.feature_extractor import FeatureExtractor
from sample.debug import Debug
from tests.conftest import example_vid


def test_load_image(test_image):
    assert len(test_image)


@pytest.mark.skip(reason="GUI")
def test_display_image(test_image):
    Debug.display_image(test_image)


def test_get_histograms_rgb(test_image):
    hists = Debug.get_histograms_rgb(test_image)
    for hist in hists:
        assert len(hist)


@pytest.mark.skip(reason="GUI")
def test_plot_histogram_rgb(test_image):
    hists = Debug.get_histograms_rgb(test_image)
    Debug.plot_histograms(hists)


@pytest.mark.skip(reason="GUI")
def test_plot_histogram_hsv(test_image):
    hists = FeatureExtractor.get_histograms_hsv(test_image)
    Debug.plot_histograms(hists)


@pytest.mark.skip(reason="GUI")
def test_display_video_histogram_hsv():
    Debug.plot_histograms_live(example_vid)
    # TODO Fail if not displayed
