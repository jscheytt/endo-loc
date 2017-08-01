import argparse

import helper.helper as hlp
import sample.sample as s
import prep.preprocessor as pre


def main(dir_train, dir_eval, clf_filepath, C_value, gamma_value):
    hlp.setup_logging()

    global C, gamma
    if C_value is not (None or ""):
        try:
            C = float(C_value)
        except ValueError:
            pass
    if gamma_value is not (None or ""):
        try:
            gamma = float(gamma_value)
        except ValueError:
            pass
    if C_value is not (None or "") and gamma_value is not (None or ""):
        do_grid_search = False
    else:
        do_grid_search = True

    X, y = pre.get_multiple_data_and_targets(dir_filepath=dir_train, do_subsampling=True)
    if do_grid_search:
        best_parameters = s.get_best_params(X, y)
        hlp.log(best_parameters)
    else:
        best_parameters = {'C': C, 'gamma': gamma}
    classifier = s.get_svclassifier(X, y, **best_parameters)
    s.write_classifier(classifier, clf_filepath)

    X_eval, y_eval = pre.get_multiple_data_and_targets(dir_filepath=dir_eval)
    validation, conf_mat = s.get_evaluation_report(classifier, X_eval, y_eval)
    hlp.log(validation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and export a binary classifier for endoscope "
                                                 "inside/outside patient classification based on HSV histograms.")
    parser.add_argument("dir_train", help="Directory containing all feature XMLs and label CSVs for training the "
                                          "classifier. CSVs need to have the same file name as their corresponding "
                                          "XML.")
    parser.add_argument("dir_eval", help="Directory containing the feature XML(s) and label CSV(s) for evaluating "
                                         "the classifier's performance. CSV(s) must have the same file name as their "
                                         "corresponding XML.")
    parser.add_argument("clf_filepath", help="Filepath where the final classifier should be exported to.")
    parser.add_argument("--C_value", "-c", help="Omit the grid search and directly specify a C value.")
    parser.add_argument("--gamma", "-g", help="Omit the grid search and directly specify a gamma value.")
    args = parser.parse_args()
    main(args.dir_train, args.dir_eval, args.clf_filepath, C_value=args.C_value, gamma_value=args.gamma)
