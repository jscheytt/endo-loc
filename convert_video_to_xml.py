import argparse

import feature_extraction.ft_extractor as fx
import helper.helper as hlp

DEF_TARGET_FILENAME_PATTERN = "{}_{}.xml"


def get_default_target_filename(video_filename, h_c, s_c, v_c):
    prefix = hlp.strip_file_extension(video_filename)
    suffix = ""
    if h_c:
        suffix += "h"
    if s_c:
        suffix += "s"
    if v_c:
        suffix += "v"
    return DEF_TARGET_FILENAME_PATTERN.format(prefix, suffix)


def main(videofile, no_h_channel=False, no_s_channel=False, no_v_channel=False):
    hlp.setup_logging()

    h_c = not no_h_channel
    s_c = not no_s_channel
    v_c = not no_v_channel

    videoxml = fx.get_xml_from_videofile(videofile, h_c, s_c, v_c)

    target_filename = get_default_target_filename(videofile, h_c, s_c, v_c)
    fx.write_video_to_xml(videoxml, target_filename=target_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a video file to an XML containing the HSV histograms of "
                                                 "each frame.")
    parser.add_argument("videofile", help="The video file to convert")
    parser.add_argument("-c", "--no_h_channel", help="Exclude the H channel", action="store_true")
    parser.add_argument("-s", "--no_s_channel", help="Exclude the S channel", action="store_true")
    parser.add_argument("-v", "--no_v_channel", help="Exclude the V channel", action="store_true")
    args = parser.parse_args()
    main(args.videofile, args.no_h_channel, args.no_s_channel, args.no_v_channel)
