import helper.helper as hlp


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

    DEF_FPS = 25  # default framerate
    DEF_NUMPX = 1920 * 1080

    def __init__(self, fps=DEF_FPS, frames=None, labels=None, label_list=None, numpx=DEF_NUMPX, xmlpath=""):
        if labels is None:
            labels = []
        if frames is None:
            frames = []
        self.fps = int(fps)
        self.frames = frames
        self.labels = labels
        self.numpx = numpx
        self.label_list = label_list

        if xmlpath != "":
            import feature_extraction.ft_extractor as fx
            self.fps, self.frames, self.numpx = fx.get_video_params_from_xml(xmlpath)

        if self.label_list is None:
            self.label_list = []
        else:
            self.adjust_list_lengths()

    def add_frame(self, frame):
        """
        Add a vframe obj to the list of frames of a video.
        :param frame: VFrame obj to be appended
        :return:
        """
        assert isinstance(frame, VFrame)
        self.frames.append(frame)

    def get_label(self, timestamp):
        """
        Retrieve the target label corresponding to a certain timestamp from the list of labels.
        :param timestamp: Timestamp obj
        :return: ILabel obj corresponding to timestamp
        """
        for label in self.labels:
            if label.start <= timestamp < label.end:
                return label

    def fill_label_list(self):
        """
        Get all labels as an exhaustive list, i. e. with as many entries as there are frames.
        This label_list is also added as an attribute of the Video object.
        :return: 1D list of ILabel objs
        """
        if not self.label_list:
            import label_import.timestamp as ts
            last_label = self.labels[-1]
            last_timestamp = last_label.end
            last_frameidx = last_timestamp.get_frameidx(self.fps)

            for idx in range(0, last_frameidx, 1):
                curr_timestamp = ts.Timestamp.from_frameidx_fps(idx, self.fps)
                curr_label = self.get_label(curr_timestamp)
                label_val = curr_label.value.value
                self.label_list.append(label_val)

            self.adjust_list_lengths()

    def adjust_list_lengths(self):
        """
        Validate label list length. Truncate or extend if necessary.
        :return:
        """
        if self.frames:
            if len(self.label_list) > len(self.frames):
                del self.label_list[len(self.frames):]
            elif len(self.label_list) < len(self.frames):
                last_elem = self.label_list[-1]
                while len(self.label_list) < len(self.frames):
                    self.label_list.append(last_elem)
            self.discard_obsolete_frames()

    def discard_obsolete_frames(self):
        """
        Delete frames with ADS label.
        :return:
        """
        import label_import.label as l
        for idx in reversed(range(len(self.label_list))):
            if self.label_list[idx] == l.ILabelValue.ADS.value:
                del self.frames[idx]
                del self.label_list[idx]

    def get_featurevector_list(self):
        """
        Get all 1D vectors of the feature descriptors of the frames.
        :return: list of all feature vectors
        """
        vectors = [f.descriptor.get_vector() for f in self.frames]
        return vectors

    def write_label_list(self, filename):
        """
        Write label list to a CSV file.
        :param filename: file to write to
        :return:
        """
        import csv
        from debug.debug import LogCont
        with LogCont("Write label list to CSV file"):
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=hlp.VAL_SEP, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                self.fill_label_list()
                for idx, label in enumerate(self.label_list):
                    writer.writerow([self.frames[idx].timestamp.to_str(), label])
