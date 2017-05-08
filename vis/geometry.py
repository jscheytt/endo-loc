import cv2
import math

import vis.display as dsp


def resize_img(img, fx=.5, fy=.5, interpolation=cv2.INTER_LINEAR):
    """
    Resize an image so e. g. it can be displayed fully on the screen.
    :param img: 
    :param fx: scaling factor in x
    :param fy: scaling factor in y
    :param interpolation: 
    :return: 
    """
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)


def resize_for_fullscreen(img):
    """
    Resize an image to fullscreen, i. e.: rescale it to fit screen
    (both images with shape > or < screen dimensions),
    fill rest of image with black.
    :param img: 
    :return: img rescaled and filled with black
    """
    screen_width, screen_height = dsp.get_screen_dims()
    img_width, img_height = get_img_dims(img)

    ratio_screen = screen_width / screen_height
    ratio_img = img_width / img_height
    ratio_of_ratios = ratio_screen / ratio_img

    if ratio_of_ratios >= 1.0:
        new_height = screen_height
        scaling = new_height / img_height
    else:
        new_width = screen_width
        scaling = new_width / img_width
    resized = resize_img(img, fx=scaling, fy=scaling)

    res_width, res_height = get_img_dims(resized)
    if ratio_of_ratios >= 1.0:
        border_top = border_bottom = 0
        border_left = math.floor((screen_width - res_width) / 2)
        border_right = math.ceil((screen_width - res_width) / 2)
    else:
        border_top = math.floor((screen_height - res_height) / 2)
        border_bottom = math.ceil((screen_height - res_height) / 2)
        border_left = border_right = 0
    filled = cv2.copyMakeBorder(resized, border_top, border_bottom, border_left, border_right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return filled


def get_img_dims(img):
    """
    Get width and height of an image.
    :param img: 
    :return: tuple(width, height)
    """
    height, width = img.shape[:2]
    return width, height
