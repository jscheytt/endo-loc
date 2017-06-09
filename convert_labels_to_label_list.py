import argparse

import feature_extraction.ft_descriptor as fd
import label_import.label_importer as li
import helper.helper as hlp

DEF_TARGET_FILENAME_PATTERN = "{}.csv"


def get_default_target_filename(filename):
    prefix = hlp.strip_file_extension(filename)
    return DEF_TARGET_FILENAME_PATTERN.format(prefix)


def main(labelsfile, featurefile):
    hlp.setup_logging()

    labels = li.read_labels(labelsfile)
    target_filename = get_default_target_filename(labelsfile)

    video = fd.Video(xmlpath=featurefile, labels=labels)
    video.write_label_list(target_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a labels file to a CSV file "
                                                 "containing the HSV histograms of each frame.")
    parser.add_argument("labelsfile", help="The labels file (.ass) to be converted")
    parser.add_argument("featurefile", help="The video feature file (.xml) corresponding to the labels")
    args = parser.parse_args()
    main(args.labelsfile, args.featurefile)
