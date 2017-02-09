import cv2
import logging
from lxml import etree

import feature_extraction.ft_descriptor as fd
import helper.helper as hlp
import label_import.timestamp as lt

CHARSET = "utf-8"
VIDEO_TAG = "video"
VFRAME_TAG = "vframe"
TIMESTAMP_TAG = "timestamp"
FTDESCR_TAG = "ftdescr"
HIST_TAG = "hist"
FPS_TAG = "fps"
NUMPX_TAG = "numpx"
VAL_SEP = ';'


def get_histograms_hsv(image):
    """
    Retrieve HSV histogram of an image.
    :param image: OpenCV image obj
    :return: List of histograms in order H, S, V
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = hlp.flatten_int(cv2.calcHist([image_hsv], [0], None, [180], [0, 180]))
    hist_s = hlp.flatten_int(cv2.calcHist([image_hsv], [1], None, [256], [0, 256]))
    hist_v = hlp.flatten_int(cv2.calcHist([image_hsv], [2], None, [256], [0, 256]))
    return [hist_h, hist_s, hist_v]


def get_descriptor_as_xml(descriptor):
    """
    Get XML element from a FeatureDescriptor obj
    :param descriptor: FeatureDescriptor obj
    :return: XML element of the FeatureDescriptor obj
    """
    descriptor_el = etree.Element(FTDESCR_TAG)
    for idx, hist in enumerate(descriptor.hists):
        hist_el = etree.SubElement(descriptor_el, HIST_TAG)
        hist_el.text = VAL_SEP.join(str(item) for item in hist)
        descriptor_el.append(hist_el)
    return descriptor_el


def get_vframe_as_xml(frame):
    """
    Get XML element from a VFrame obj
    :param frame: VFrame obj
    :return: XML element of the VFrame obj
    """
    frame_el = etree.Element(VFRAME_TAG)
    frame_el.set(TIMESTAMP_TAG, frame.timestamp.to_str())
    descriptor_el = get_descriptor_as_xml(frame.descriptor)
    frame_el.append(descriptor_el)
    return frame_el


def get_xml_from_videofile(filename):
    """
    Read in a video file and build an XML element from it.
    :param filename: Path to video file
    :return: XML element of the video file
    """
    cap = cv2.VideoCapture(filename)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    numpx = 0
    curr_frameidx = 0

    video_el = etree.Element(VIDEO_TAG)
    video_el.set(FPS_TAG, str(fps))
    # video = fd.Video(fps=fps)
    # TODO get_video_as_xml

    while cap.isOpened():
        _, frame_img = cap.read()

        if frame_img is not None:
            # TODO reject frames with reduced label None
            # Initialize numpx
            if numpx == 0:
                numpx = get_img_numpx(frame_img)

            curr_timestamp = lt.Timestamp.from_frameidx_fps(curr_frameidx, fps)
            vframe = fd.VFrame(curr_timestamp, fd.FeatureDescriptor(frame_img))
            # video.add_frame(vframe)

            vframe_el = get_vframe_as_xml(vframe)
            video_el.append(vframe_el)

            # Log timestamp for every second
            if curr_frameidx % fps == 0:
                logging.info(curr_timestamp.to_str())

            curr_frameidx += 1
        else:
            # End of file reached
            break

    video_el.set(NUMPX_TAG, str(numpx))
    cap.release()

    return video_el


def write_video_to_xml(video_xml, target_filename):
    """
    Write Video XML element to an XML file.
    :param video_xml: Video XML element
    :param target_filename: Path to target XML file
    :return:
    """
    tree = etree.ElementTree(video_xml)
    tree.write(target_filename, encoding=CHARSET, pretty_print=True)


def get_video_from_xml(filename):
    """
    Retrieve Video obj from Video XML.
    :param filename: Path to video XML file
    :return: Video obj from XML file
    """
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

        timestamp_str = vframe_el.get(TIMESTAMP_TAG)
        timestamp_obj = lt.Timestamp.from_str(timestamp_str)
        vframe = fd.VFrame(timestamp=timestamp_obj, descriptor=descriptor)
        frames.append(vframe)

    video = fd.Video(fps=video_el.get(FPS_TAG), frames=frames, numpx=video_el.get(NUMPX_TAG))

    return video


def get_img_numpx(img):
    """
    Get total number of pixels in an image.
    :param img: OpenCV image obj
    :return: Number of pixels in img
    """
    height, width = tuple(img.shape[1::-1])
    return height * width


def load_image(filepath):
    """
    Load color image.
    :param filepath: Path to image file
    :return: OpenCV image object
    """
    return cv2.imread(filepath, cv2.IMREAD_COLOR)
