from enum import Enum


class ILabelValue(Enum):
    """
    Possible values for an ILabel.
    """
    IN, OUT, MOVING_IN, MOVING_OUT, IN_BETWEEN, EXIT, ADS = range(7)


class ILabel:
    """
    Label denoting location of the endoscope.
    """

    def __init__(self, start, end, value):
        """
        Build an ILabel obj.
        :param start: Timestamp obj for start of segment containing label
        :param end: Timestamp obj for end of segment containing label
        :param value: ILabelValue obj
        """
        self.start = start
        self.end = end
        self.value = value

    def __len__(self):
        return len(self.start) + len(self.end) + self.value.value


