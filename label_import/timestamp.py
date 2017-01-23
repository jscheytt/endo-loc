MS_PER_S = 1000
S_PER_MIN = 60
MIN_PER_H = 60


class Timestamp:
    """
    Representation of a timestamp.
    """
    PATTERN = "%02d:%02d:%02d.%02d"

    def __init__(self, hours, minutes, seconds, milliseconds):
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.milliseconds = milliseconds
        "Representation of the timestamp as total sum of milliseconds"
        self.total_ms = self.milliseconds \
                        + self.seconds * MS_PER_S \
                        + self.minutes * S_PER_MIN * MS_PER_S \
                        + self.hours * MIN_PER_H * S_PER_MIN * MS_PER_S

    @classmethod
    def from_str(cls, timestamp_str):
        """
        Construct Timestamp from timestamp str.
        :param timestamp_str:
        :return:
        """
        units = timestamp_str.split(":")
        seconds_ms = units[-1].split(".")
        hours = int(units[0])
        minutes = int(units[1])
        seconds = int(seconds_ms[0])
        milliseconds = int(seconds_ms[1])
        return cls(hours, minutes, seconds, milliseconds)

    @classmethod
    def from_frameidx_fps(cls, frameidx, fps):
        """
        Construct Timestamp from frame index and fps
        :param frameidx:
        :param fps: Frames per second
        :return:
        """
        import helper.helper as hlp
        milliseconds = (frameidx % fps) * 100 / fps
        total_seconds = frameidx // fps
        seconds = total_seconds % S_PER_MIN
        total_minutes = total_seconds // S_PER_MIN
        minutes = total_minutes % MIN_PER_H
        hours = hlp.clamp(total_minutes // MIN_PER_H, 0, 99)
        return cls(hours, minutes, seconds, milliseconds)

    def to_str(self):
        """
        Get str representation of Timestamp obj.
        :return: str representation of Timestamp obj
        """
        return self.PATTERN % (self.hours, self.minutes, self.seconds, self.milliseconds)

    def get_frameidx(self, fps):
        """
        Get index of the frame corresponding to this Timestamp.
        :param fps: Frames per second
        :return: Frame index of this Timestamp
        """
        return self.hours * MIN_PER_H * S_PER_MIN * fps \
               + self.minutes * S_PER_MIN * fps \
               + self.seconds * fps \
               + self.milliseconds // (100 / fps)

    def __len__(self):
        return len(self.to_str())

    def __eq__(self, other):
        return self.total_ms == other.total_ms

    def __lt__(self, other):
        return self.total_ms < other.total_ms

    def __gt__(self, other):
        return self.total_ms > other.total_ms

    def __le__(self, other):
        return self.total_ms <= other.total_ms

    def __ge__(self, other):
        return self.total_ms >= other.total_ms
