import feature_extraction.ft_extractor as fx

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
            self.hists = fx.get_histograms_hsv(hists_or_img)

    def get_vector(self):
        return self.hists[0] + self.hists[1] + self.hists[2]


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

    def __init__(self, fps=DEF_FPS, frames=None, labels=None, numpx=DEF_NUMPX, xmlpath=""):
        if labels is None:
            labels = []
        if frames is None:
            frames = []
        self.fps = fps
        self.frames = frames
        self.labels = labels
        self.numpx = numpx

        if xmlpath != "":
            self.fps, self.frames, self.numpx = fx.get_video_params_from_xml(xmlpath)

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

    def get_label_list(self):
        """
        Get all labels as an exhaustive list, i. e. with as many entries as there are frames.
        :return: 1D list of ILabel objs
        """
        labels = []
        for frame in self.frames:
            label_val = self.get_label_from_timestamp(frame.timestamp).value.value
            labels.append(label_val)
        return labels

    def get_featurevector_list(self):
        """
        Get all 1D vectors of the feature descriptors of the frames.
        :return: list of all feature vectors
        """
        vectors = [f.descriptor.get_vector() for f in self.frames]
        return vectors
