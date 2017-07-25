import argparse
import os

import prep.preprocessor as pre
import sample.sample as s
import helper.helper as hlp
import debug.debug as dbg


def main(dir_train_or_clf, dir_eval, do_subsampling, dir_y):
    hlp.setup_logging()
    svc = None
    evaluation = "Please provide at least one directory with files!"
    conf_mat = None

    if not os.path.isdir(dir_train_or_clf):  # Load classifier
        svc = s.read_classifier(dir_train_or_clf)
    else:  # Or create it
        X, y = pre.get_multiple_data_and_targets(dir_filepath=dir_train_or_clf, do_subsampling=do_subsampling)
        if dir_eval is not None:  # Learn and evaluate it
            svc = s.get_svclassifier(X, y)
        else:  # Or do cross validation
            evaluation, y_pred = s.get_crossval_evaluation(X, y, print_scores=True)
            if dir_y is not None:
                # Write y_true and y_pred to disk
                textfile = dir_y + os.sep + "predicted.txt"
                eval_labels_file = dir_y + os.sep + "y_eval.txt"
                dbg.write_list_to_file(y_pred, textfile)
                dbg.write_list_to_file(y, eval_labels_file)

    if dir_eval is not None and svc is not None:
        X_eval, y_eval = pre.get_multiple_data_and_targets(dir_filepath=dir_eval)
        predicted = s.get_prediction(X_eval, svc)
        evaluation, conf_mat = s.get_evaluation_report(svc, X_eval, y_eval, predicted=predicted)

        # Write y_eval and predicted to disk
        textfile = dir_eval + os.sep + "predicted.txt"
        eval_labels_file = dir_eval + os.sep + "y_eval.txt"
        dbg.write_list_to_file(predicted, textfile)
        dbg.write_list_to_file(y_eval, eval_labels_file)

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
    parser.add_argument("-de", "--dir_eval",
                        help="Directory containing the feature XML(s) and label CSV(s) for "
                             "evaluating the classifier's performance. CSV(s) must have the "
                             "same file name as their corresponding XML. Skip if you want to "
                             "perform cross validation.")
    parser.add_argument("-dy", "--dir_y",
                        help="Directory for storing the true and predicted target values.")
    parser.add_argument("-s", "--subsampling", help="Subsample majority class", action="store_true")
    args = parser.parse_args()
    main(args.dir_train_or_clf, args.dir_eval, args.subsampling, args.dir_y)
