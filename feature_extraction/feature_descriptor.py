DEF_FPS = 25  # default framerate
DEF_NUMPX = 1920*1080


class FeatureDescriptor:
    def __init__(self, hists_or_img):
        if isinstance(hists_or_img, list):
            self.hists = hists_or_img
        else:
            from feature_extraction.feature_extractor import FeatureExtractor
            self.hists = FeatureExtractor.get_histograms_hsv(hists_or_img)


class VFrame:
    def __init__(self, timestamp="", descriptor=None):
        self.timestamp = timestamp
        self.descriptor = descriptor


class Video:
    def __init__(self, fps=DEF_FPS, frames=None, labels=None, numpx=DEF_NUMPX):
        if labels is None:
            labels = []
        if frames is None:
            frames = []
        self.fps = fps
        self.frames = frames
        self.labels = labels
        self.numpx = numpx

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_label_from_timestamp(self, t):
        for label in self.labels:
            if label.start <= t < label.end:
                return label
