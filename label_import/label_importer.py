import label_import.label as lb
import label_import.timestamp as lt

LINE_PATTERN = r"Dialogue: 0,([0-9.:]*?),([0-9.:]*?),Default,,0,0,0,,([A-Z_]*)"


def get_textfile_as_str(ilabel_file):
    """
    Open textfile and return contents as one multi-line str.
    :param ilabel_file: Path to textfile
    :return: Str containing all file contents
    """
    with open(ilabel_file, 'r') as file:
        contents = file.read()
    return contents


def reduce_label_value(label_value):
    """
    For binary classification, reduce the 7 labels to only 2.
    :param label_value: ILabelValue obj
    :return: ILabelValue obj of IN or OUT only
    """
    switcher = {
        lb.ILabelValue.ADS: None,
        lb.ILabelValue.MOVING_IN: lb.ILabelValue.IN,
        lb.ILabelValue.MOVING_OUT: lb.ILabelValue.IN,
        lb.ILabelValue.IN_BETWEEN: lb.ILabelValue.IN,
        lb.ILabelValue.EXIT: lb.ILabelValue.IN,
    }
    return switcher.get(label_value, label_value)


def get_label_from_line(line):
    """
    Match pattern against line and get ILabel obj from it.
    :param line: str containing pattern
    :return: ILabel obj if matching, None if not
    """
    import re
    matcher = re.match(LINE_PATTERN, line)
    ilabel = None

    if matcher is not None:
        start = lt.Timestamp.from_str(matcher.group(1))
        end = lt.Timestamp.from_str(matcher.group(2))
        label_name = matcher.group(3)
        label_idx = [l.name for l in lb.ILabelValue].index(label_name)
        label_value = [l for l in lb.ILabelValue][label_idx]
        redu_label_val = reduce_label_value(label_value)
        ilabel = lb.ILabel(start, end, redu_label_val)

    return ilabel


def get_labels_from_mlstring(file_cont):
    """
    Retrieve all the labels inside one multi-line str.
    :param file_cont: Str containing lines with labels acc. to pattern
    :return: List of ILabel objs
    """
    ilabels = []
    for line in file_cont.splitlines():
        ilabel = get_label_from_line(line)
        if ilabel is not None:
            ilabels.append(ilabel)
    return ilabels


def read_labels(filename):
    """
    Retrieve all labels from a textfile.
    :param filename: Path to textfile
    :return: List of ILabel objs
    """
    file_cont = get_textfile_as_str(filename)
    ilabels = get_labels_from_mlstring(file_cont)
    return ilabels


def read_label_list(filename):
    """
    Read a list of label values from a CSV file.
    :param filename: Path to the CSV file
    :return: 1D list of label values
    """
    import csv
    import helper.helper as hlp
    label_list = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=hlp.VAL_SEP, quotechar='|')
        for row in reader:
            label_list.append(row[1])
    return label_list
