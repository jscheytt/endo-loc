import cv2
from lxml import etree

import feature_extraction.feature_descriptor as fd
import feature_extraction.helper as hlp
import tests.conftest as cft

# Constants
CHARSET = "utf-8"
VIDEO_TAG = "video"
VFRAME_TAG = "vframe"
TIMESTAMP_TAG = "timestamp"
FTDESCR_TAG = "ftdescr"
HIST_TAG = "hist"
FPS_TAG = "fps"
VAL_SEP = ';'


class FeatureExtractor:
    @classmethod
    def get_histograms_hsv(cls, image):
        """
        Retrieve HSV histogram of an image.
        :param image:
        :return: List of histograms in order H, S, V.
        """
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = hlp.flatten_int(cv2.calcHist([image_hsv], [0], None, [180], [0, 180]))
        hist_s = hlp.flatten_int(cv2.calcHist([image_hsv], [1], None, [256], [0, 256]))
        hist_v = hlp.flatten_int(cv2.calcHist([image_hsv], [2], None, [256], [0, 256]))
        return [hist_h, hist_s, hist_v]

    @classmethod
    def get_descriptor_as_xml(cls, descriptor):
        descriptor_el = etree.Element(FTDESCR_TAG)
        for idx, hist in enumerate(descriptor.hists):
            hist_el = etree.SubElement(descriptor_el, HIST_TAG)
            hist_el.text = VAL_SEP.join(str(item) for item in hist)
            descriptor_el.append(hist_el)
        return descriptor_el

    @classmethod
    def get_vframe_as_xml(cls, frame):
        frame_el = etree.Element(VFRAME_TAG)
        frame_el.set(TIMESTAMP_TAG, str(frame.timestamp))
        # frame_el.set("timestamp", str(frame.timestamp))
        descriptor_el = cls.get_descriptor_as_xml(frame.descriptor)
        frame_el.append(descriptor_el)
        return frame_el

    @classmethod
    def get_videofile_as_xml(cls, filename):
        cap = cv2.VideoCapture(filename)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        curr_frameidx = 0

        video_el = etree.Element(VIDEO_TAG)
        video_el.set(FPS_TAG, str(fps))
        # video = fd.Video(fps=fps)
        # TODO get_video_as_xml

        while cap.isOpened():
            _, frame_img = cap.read()

            if frame_img is not None:
                curr_timestamp = hlp.get_timestamp(curr_frameidx, fps)
                vframe = fd.VFrame(curr_timestamp, fd.FeatureDescriptor(frame_img))
                # video.add_frame(vframe)

                vframe_el = cls.get_vframe_as_xml(vframe)
                video_el.append(vframe_el)

                curr_frameidx += 1
            else:
                # End of file reached
                break

        cap.release()
        cv2.destroyAllWindows()

        return video_el

    @classmethod
    def write_video_to_xml(cls, video_xml):
        tree = etree.ElementTree(video_xml)
        tree.write(cft.test_video_xml_filename, encoding=CHARSET, pretty_print=True)

    @classmethod
    def get_video_from_xml(cls, filename):
        parser = etree.XMLParser(encoding=CHARSET)
        tree = etree.parse(filename, parser=parser)
        video_el = tree.getroot()
        frames_els = video_el.getchildren()
        frames = []

        for vframe_el in frames_els:
            descriptor_el = vframe_el[0]
            hists = []
            for hist_el in descriptor_el:
                hist = [int(x) for x in hist_el.text.split(VAL_SEP)]
                hists.append(hist)
            descriptor = fd.FeatureDescriptor(hists)

            timestamp = vframe_el.get(TIMESTAMP_TAG)
            vframe = fd.VFrame(timestamp=timestamp, descriptor=descriptor)
            frames.append(vframe)

        video = fd.Video(fps=video_el.get(FPS_TAG), frames=frames)

        return video
