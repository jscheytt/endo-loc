import argparse

import prep.preprocessor as pre
import sample.sample as s
import helper.helper as hlp


def main(dir_train, dir_eval, do_subsampling):
    hlp.setup_logging()

    X, y = pre.get_multiple_data_and_targets(dir_filepath=dir_train, do_subsampling=do_subsampling)

    if dir_eval is not None:
        svc = s.get_svclassifier(X, y)
        X_eval, y_eval = pre.get_multiple_data_and_targets(dir_filepath=dir_eval)
        evaluation = s.get_evaluation_report(svc, X_eval, y_eval)
    else:
        evaluation = s.get_crossval_evaluation(X, y, print_scores=True)

    print(evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print evaluation metrics for learning an HSV classifier "
                                                 "on 1 or 2 sets of feature data + targets.")
    parser.add_argument("dir_train", help="Directory containing all feature XMLs and label CSVs for training the "
                                          "classifier. CSVs need to have the same file name as their corresponding "
                                          "XML.")
    parser.add_argument("-de", "--dir_eval", help="Directory containing the feature XML(s) and label CSV(s) for "
                                                  "evaluating the classifier's performance. CSV(s) must have the "
                                                  "same file name as their corresponding XML. Skip if you want to "
                                                  "perform cross validation.")
    parser.add_argument("-s", "--subsampling", help="Subsample majority class", action="store_true")
    args = parser.parse_args()
    main(args.dir_train, args.dir_eval, args.subsampling)
