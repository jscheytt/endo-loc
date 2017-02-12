import argparse
import logging
import logging.config

import feature_extraction.ft_extractor as fx
import helper.helper as hlp

DEF_TARGET_FILENAME_PATTERN = "{}_ft.xml"


def get_default_target_filename(video_filename):
    prefix = hlp.strip_file_extension(video_filename)
    return DEF_TARGET_FILENAME_PATTERN.format(prefix)


def main(videofile):
    # Load Logging settings
    logging.config.fileConfig("logging_config.ini")

    videoxml = fx.get_xml_from_videofile(videofile)

    target_filename = get_default_target_filename(videofile)
    fx.write_video_to_xml(videoxml, target_filename=target_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a video file to an XML containing the HSV histograms of "
                                                 "each frame.")
    parser.add_argument("videofile", help="The video file to be converted")
    args = parser.parse_args()
    main(args.videofile)
