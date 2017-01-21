class Timestamp:
    def __init__(self, timestamp_str):
        import feature_extraction.helper as hlp
        self.hours, self.minutes, self.seconds, self.milliseconds = hlp.get_time_from_timestamp(timestamp_str)
        self.tostring = hlp.get_timestamp_from_time(self.hours, self.minutes, self.seconds, self.milliseconds)
        self.total_ms = self.milliseconds \
                        + self.seconds * hlp.MS_PER_S \
                        + self.minutes * hlp.S_PER_MIN * hlp.MS_PER_S \
                        + self.hours * hlp.MIN_PER_H * hlp.S_PER_MIN * hlp.MS_PER_S

    def __len__(self):
        return len(self.tostring)

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
