import cv2

import feature_extraction.ft_descriptor as fd
import feature_extraction.ft_extractor as fx
import label_import.label as ll
import prep.preprocessor as pre
import sample.sample as s
import vis.display as dsp
import vis.geometry as geom

CLF = None
PREV_LABEL = None


def display_predict_on_frame(frame, skip_frames=0, predict_downscaled=True, display_downscaled=False,
                             h_c=True, s_c=True, v_c=True):
    """
    Display a video stream and classify each frame live.
    :param frame:
    :param skip_frames:
    :param predict_downscaled: predict on a downscaled version of the frame
    :param display_downscaled: display a downscaled version of the frame
    :param h_c: predict on hue channel
    :param s_c: predict on saturation channel
    :param v_c: predict on value channel
    :return: 
    """
    global PREV_LABEL
    label = PREV_LABEL
    prepped = geom.fill_img_for_fullscreen(frame)
    dst = prepped

    if skip_frames == 0 or (skip_frames > 0 and dsp.FRAME_COUNT % skip_frames == 0):
        downscaled = None
        if predict_downscaled:
            downscaled = geom.resize_img(frame, fx=0.4, fy=0.4)
            frame = downscaled
        ft_vec = get_live_ft_vec(frame, h_c=h_c, s_c=s_c, v_c=v_c)
        label = predict_label(CLF, ft_vec)
        if skip_frames > 0 and dsp.FRAME_COUNT % skip_frames == 0:
            PREV_LABEL = label
        if display_downscaled:
            prepped = geom.fill_img_for_fullscreen(downscaled)
            draw_label(prepped, label)

    draw_label(prepped, label)
    draw_menu(prepped)

    dsp.show_frame(dst, fullscreen=True)


def get_live_ft_vec(img, h_c=True, s_c=True, v_c=True):
    """
    Directly get feature vector of an image.
    :param img:
    :param h_c: predict on hue channel
    :param s_c: predict on saturation channel
    :param v_c: predict on value channel
    :return: normalized feature vector
    """
    histograms = fx.get_histograms_hsv(img)
    ft_desc = fd.FeatureDescriptor(histograms, h_c, s_c, v_c)
    ft_vec = ft_desc.get_vector()
    ft_vec_np = pre.get_array(ft_vec)
    max_val = fx.get_img_numpx(img)
    ft_vec_norm = pre.get_norm_ft_vec(ft_vec_np, float(max_val), aslist=False)
    return ft_vec_norm


def predict_label(classifier, ft_vec):
    """
    Predict the class label of the incoming feature vector
    based on the input classifier.
    :param classifier: sklearn classifier 
    :param ft_vec: normalized feature vector
    :return: ILabelValue.IN or .OUT
    """
    value = s.predict_single_ft_vec(classifier, ft_vec)
    return ll.ILabelValue(value)


def draw_label(img, label):
    """
    Draw a text showing which label has been detected.
    :param img: image to be drawn on
    :param label: ILabelValue
    :return: 
    """
    width, height = geom.get_img_dims(img)

    font_scale = 0.003 * height
    position = (int(0.04 * width), int(0.12 * height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness_inline = int(3 * font_scale)
    line_type = cv2.LINE_AA

    color_outline = (0, 0, 0)
    thickness_outline = int(2.5 * 3 * font_scale)

    if label is ll.ILabelValue.IN:
        text = "INSIDE"
        color_inline = (0, 255, 0)
    else:
        text = "OUTSIDE"
        color_inline = (255, 255, 255)

    cv2.putText(img, text, position, font, font_scale, color_outline, thickness_outline, line_type)
    cv2.putText(img, text, position, font, font_scale, color_inline, thickness_inline, line_type)


def draw_menu(img):
    """
    Draw the menu. So far this is only a text in the right corner about 'Q for quit'.
    :param img: image to be drawn on
    :return: 
    """
    width, height = geom.get_img_dims(img)

    font_scale = 0.0012 * height
    position = (int(0.8 * width), int(0.07 * height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = int(2 * font_scale)
    line_type = cv2.LINE_AA
    text = "Press 'Q' to quit"
    color = (255, 255, 255)

    cv2.putText(img, text, position, font, font_scale, color, thickness, line_type)
