DEF_FPS = 25  # default framerate
DEF_NUMPX = 1920 * 1080


class FeatureDescriptor:
    """
    A feature descriptor containing the 3 histogram channels as a list.
    """

    def __init__(self, hists_or_img):
        if isinstance(hists_or_img, list):
            self.hists = hists_or_img
        else:
            import feature_extraction.ft_extractor as fx
            self.hists = fx.get_histograms_hsv(hists_or_img)


class VFrame:
    """
    Representation of a video frame. Contains the timestamp and the feature descriptor.
    """

    def __init__(self, timestamp=None, descriptor=None):
        self.timestamp = timestamp
        if timestamp is not None:
            from label_import.timestamp import Timestamp
            assert isinstance(timestamp, Timestamp)
        self.descriptor = descriptor
        if descriptor is not None:
            assert isinstance(descriptor, FeatureDescriptor)


class Video:
    """
    Representation of an entire video file. FPS and the total number of pixels is defined.
    All frames and corresponding target labels are contained herein.
    """

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
        """
        Add a vframe obj to the list of frames of a video.
        :param frame: VFrame obj to be appended
        :return:
        """
        assert isinstance(frame, VFrame)
        self.frames.append(frame)

    def get_label_from_timestamp(self, timestamp):
        """
        Retrieve the target label corresponding to a certain timestamp from the list of labels.
        :param timestamp: Timestamp obj
        :return: ILabel obj corresponding to timestamp
        """
        for label in self.labels:
            if label.start <= timestamp < label.end:
                return label
