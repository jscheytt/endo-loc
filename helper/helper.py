import logging
import logging.config
import os

import cv2
import yaml

VAL_SEP = ';'


def clamp(n, minn, maxn):
    """
    Clamp an integer to a certain range.
    :param n: int to be clamped
    :param minn: lower bound
    :param maxn: upper bound
    :return: clamped int
    """
    return min(max(n, minn), maxn)


def flatten_int(l):
    """
    Flatten a multi-dimensional list to a one-dimensional and convert all values to integers.
    :param l: list of lists with values that can be cast to int
    :return: flattened int list
    """
    return [int(item) for sublist in l for item in sublist]


def file_length(filename):
    """
    Get byte length of a file.
    :param filename: Path to file
    :return: Byte length of file
    """
    try:
        f = open(filename)
        ret = int(os.fstat(f.fileno()).st_size)
    except FileNotFoundError:
        ret = -1
    return ret


def get_full_path_from_projroot(filename):
    """
    Retrieve the full path to filename. Assumes this function is called from inside a module.
    :param filename:
    :return: full filepath to filename
    """
    cwd = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(cwd, "..")
    return os.path.join(root, filename)


def xml_elements_equal(e1, e2):
    """
    Compare 2 XML elements by content.
    :param e1: first XML element
    :param e2: second XML element
    :return: True if two xml elements are the same by content
    """
    if e1.tag != e2.tag:
        return False
    if e1.text != e2.text:
        return False
    if e1.tail != e2.tail:
        return False
    if e1.attrib != e2.attrib:
        return False
    if len(e1) != len(e2):
        return False
    return all(xml_elements_equal(c1, c2) for c1, c2 in zip(e1, e2))


def strip_file_extension(filename):
    return filename.rsplit('.', 1)[0]


def maxval_of_2dlist(ll):
    """
    Get maximum value in a list of lists.
    :param ll: 2D list
    :return: max value in the 2D list
    """
    maxval = 0
    for l in ll:
        maxval = max(l)
    return maxval


def reverse_enum(l):
    """
    Generator for reverse traversal with access to the index.
    :param l: 
    :return: 
    """
    for index in reversed(range(len(l))):
        yield index, l[index]


def setup_logging():
    """
    Set up logging with the logging.conf file in the project root.
    :return:
    """
    os.makedirs(get_full_path_from_projroot("logs"), exist_ok=True)  # Create logging directory
    logging.config.dictConfig(yaml.load(open(get_full_path_from_projroot("logging.conf"), 'r')))


def log(message):
    """
    Log an info message to the standard loggers.
    :param message: 
    :return: 
    """
    logging.info(message)


def get_histogram(img):
    """
    Get histogram of a color or grayscale image.
    :param img: 
    :return: 
    """
    if len(img.shape) > 2:  # Color image
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
    else:  # Supposed grayscale image
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist


def compare_imgs_by_hist(img1, img2):
    """
    Compare two images by their histograms.
    The histograms are compared by their correlation.
    :param img1: 
    :param img2: 
    :return: 1.0 if images are identical, <1.0 if not, 0 if histograms
    have different shapes (e. g. because of gray to RGB comparison)
    """
    hist1 = get_histogram(img1)
    hist2 = get_histogram(img2)
    if hist1.shape != hist2.shape:
        return 0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def imgs_different(img1, img2):
    """
    Compares two images by their histogram correlation.
    :param img1: 
    :param img2: 
    :return: True if different, False if identical
    """
    return compare_imgs_by_hist(img1, img2) != 1.0


def get_binary_bool_permutations(n):
    """
    Get the full boolean permutations of n steps.
    :param n: number of variables
    :return:
    """
    permutations = []
    for i in range(2 ** n):
        permutations.append([x == '1' for x in format(i, '0' + str(n) + 'b')])
    return permutations