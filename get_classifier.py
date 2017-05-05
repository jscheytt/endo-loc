import argparse

import helper.helper as hlp
import sample.sample as s
import prep.preprocessor as pre


def main(dir_train, dir_eval, clf_filepath):
    hlp.setup_logging()

    X, y = pre.get_multiple_data_and_targets(dir_filepath=dir_train, do_subsampling=True)
    best_parameters = s.get_best_params(X, y)
    classifier = s.get_svclassifier(X, y, best_parameters)
    s.write_classifier(classifier, clf_filepath)

    X_eval, y_eval = pre.get_multiple_data_and_targets(dir_filepath=dir_eval)
    validation = s.get_evaluation_report(classifier, X_eval, y_eval)
    hlp.log(validation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and export a binary classifier for endoscope "
                                                 "inside/outside patient classification based on HSV histograms.")
    parser.add_argument("dir_train", help="Directory containing all feature XMLs and label CSVs for training the "
                                          "classifier. CSVs need to have the same file name as their corresponding "
                                          "XML.")
    parser.add_argument("dir_eval", help="Directory containing the feature XML(s) and label CSV(s) for evaluating the "
                                         "classifier's performance. CSV(s) must have the same file name as their "
                                         "corresponding XML.")
    parser.add_argument("clf_filepath", help="Filepath where the final classifier should be exported to.")
    args = parser.parse_args()
    main(args.dir_train, args.dir_eval, args.clf_filepath)
