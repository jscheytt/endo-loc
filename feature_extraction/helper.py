MS_PER_S = 1000
S_PER_MIN = 60
MIN_PER_H = 60


def clamp(n, minn, maxn):
    """
    Clamp an integer to a certain range.
    :param n:
    :param minn:
    :param maxn:
    :return:
    """
    return min(max(n, minn), maxn)


def get_timestamp(number, fps):
    milliseconds = (number % fps) * 100 / fps
    total_seconds = number // fps
    seconds = total_seconds % S_PER_MIN
    total_minutes = total_seconds // S_PER_MIN
    minutes = total_minutes % MIN_PER_H
    hours = clamp(total_minutes // MIN_PER_H, 0, 99)
    return get_timestamp_from_time(hours, minutes, seconds, milliseconds)


def get_frameidx(timestamp, fps):
    hours, minutes, seconds, milliseconds = get_time_from_timestamp(timestamp)

    frameidx = hours * MIN_PER_H * S_PER_MIN * fps \
               + minutes * S_PER_MIN * fps \
               + seconds * fps \
               + milliseconds // (100 / fps)

    return frameidx


def get_time_from_timestamp(timestamp):
    units = timestamp.split(":")
    seconds_ms = units[-1].split(".")
    hours = int(units[0])
    minutes = int(units[1])
    seconds = int(seconds_ms[0])
    milliseconds = int(seconds_ms[1])
    return hours, minutes, seconds, milliseconds


def flatten_int(l):
    """
    Flatten a multi-dimensional list to a one-dimensional and convert all values to integers.
    :param l:
    :return:
    """
    return [int(item) for sublist in l for item in sublist]


def get_timestamp_from_time(hours, minutes, seconds, milliseconds):
    return "%02d:%02d:%02d.%02d" % (hours, minutes, seconds, milliseconds)
