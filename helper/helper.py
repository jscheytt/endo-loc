import logging
import logging.config

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
    import os
    f = open(filename)
    return int(os.fstat(f.fileno()).st_size)


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
    logging.config.dictConfig(yaml.load(open('logging.conf', 'r')))
