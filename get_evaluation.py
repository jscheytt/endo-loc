import argparse
import os

import helper.helper as hlp
import prep.preprocessor as pre
import sample.sample as s
import debug.debug as dbg


def main(dir_train_or_clf, dir_eval, do_subsampling, write_labels):
    hlp.setup_logging()

    if not os.path.isdir(dir_train_or_clf):  # Load classifier
        svc = s.read_classifier(dir_train_or_clf)
    else:  # Create classifier
        X, y = pre.get_multiple_data_and_targets(dir_filepath=dir_train_or_clf, do_subsampling=do_subsampling)
        svc = s.get_svclassifier(X, y)

    X_eval, y_eval = pre.get_multiple_data_and_targets(dir_filepath=dir_eval)
    y_pred = s.get_prediction(X_eval, svc)
    evaluation, conf_mat = s.get_evaluation_report(svc, X_eval, y_eval, predicted=y_pred)

    if write_labels:
        dbg.write_list_to_dir(dir_eval, y_eval, y_pred)

    hlp.log(evaluation)
    if conf_mat is not None:
        hlp.log(conf_mat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print evaluation metrics for learning an HSV classifier "
                                                 "on 1 or 2 sets of feature data + targets.")
    parser.add_argument("dir_train_or_clf",
                        help="Directory containing all feature XMLs and label CSVs for training the "
                             "classifier. CSVs need to have the same file name as their corresponding "
                             "XML. Otherwise path to an exported classifier (PKL file).")
    parser.add_argument("dir_eval",
                        help="Directory containing the feature XML(s) and label CSV(s) for "
                             "evaluating the classifier's performance. CSV(s) must have the "
                             "same file name as their corresponding XML.")
    parser.add_argument("-s", "--subsampling", help="Subsample majority class", action="store_true")
    parser.add_argument("-wl", "--write_labels",
                        help="Write both true and predicted labels of the eval file(s) to TXT files.",
                        action="store_true")
    args = parser.parse_args()
    main(args.dir_train_or_clf, args.dir_eval, args.subsampling, args.write_labels)
