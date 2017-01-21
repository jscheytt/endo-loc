from enum import Enum
import label_import.helper as hlp


class ILabelValue(Enum):
    IN, OUT, MOVING_IN, MOVING_OUT, IN_BETWEEN, EXIT, ADS = range(7)


class ILabel:
    def __init__(self, start, end, value):
        self.start = start
        self.end = end
        self.value = value

    def __len__(self):
        return len(self.start) + len(self.end) + self.value.value


class LabelImporter:
    @classmethod
    def get_textfile_as_str(cls, ilabel_file):
        with open(ilabel_file, 'r') as file:
            contents = file.read()
        return contents

    @classmethod
    def get_label_from_line(cls, line):
        pattern = r"Dialogue: 0,([0-9.:]*?),([0-9.:]*?),Default,,0,0,0,,([A-Z_]*)"
        import re
        matcher = re.match(pattern, line)
        ilabel = None

        if matcher is not None:
            start = hlp.Timestamp(matcher.group(1))
            end = hlp.Timestamp(matcher.group(2))
            label_name = matcher.group(3)
            label_idx = [l.name for l in ILabelValue].index(label_name)
            label_value = [l for l in ILabelValue][label_idx]
            ilabel = ILabel(start, end, label_value)

        return ilabel

    @classmethod
    def get_labels_from_mlstring(cls, file_cont):
        ilabels = []
        for line in file_cont.splitlines():
            ilabel = cls.get_label_from_line(line)
            if ilabel is not None:
                ilabels.append(ilabel)
        return ilabels

    @classmethod
    def get_labels_from_file(cls, filename):
        file_cont = cls.get_textfile_as_str(filename)
        ilabels = cls.get_labels_from_mlstring(file_cont)
        return ilabels
