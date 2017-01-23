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
        ilabel = lb.ILabel(start, end, label_value)

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


def get_labels_from_file(filename):
    """
    Retrieve all labels from a textfile.
    :param filename: Path to textfile
    :return: List of ILabel objs
    """
    file_cont = get_textfile_as_str(filename)
    ilabels = get_labels_from_mlstring(file_cont)
    return ilabels
