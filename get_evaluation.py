import argparse
import logging.config

import prep.preprocessor as pre
import sample.sample as s


def main(feat_train, label_list_train, feat_eval, label_list_eval, do_subsampling):
    # Load Logging settings
    logging.config.fileConfig("logging_config.ini")

    if feat_eval is None and label_list_eval is None:
        X, y = pre.get_data_and_targets(feat_train, label_list_train, do_subsampling=do_subsampling)
        X_train, X_test, y_train, y_test = pre.get_train_test_data_targets(X, y)
    else:
        X_train, y_train = pre.get_data_and_targets(feat_train, label_list_train, do_subsampling=do_subsampling)
        X_test, y_test = pre.get_data_and_targets(feat_eval, label_list_eval)

    svc = s.get_svclassifier(X_train, y_train)
    evaluation = s.get_evaluation(svc, X_test, y_test)
    print(evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print evaluation metrics for learning an HSV classifier "
                                                 "on 2 sets of feature data + targets.")
    parser.add_argument("feat_train", help="The features file (.xml) for training the classifier")
    parser.add_argument("label_list_train", help="The label list file (.csv) for training the classifier")
    parser.add_argument("-fv", "--feat_eval", help="The features file (.xml) for evaluating the classifier")
    parser.add_argument("-lv", "--label_list_eval", help="The label list file (.csv) for evaluating the classifier")
    parser.add_argument("-s", "--subsampling", help="Subsample majority class", action="store_true")
    args = parser.parse_args()
    main(args.feat_train, args.label_list_train, args.feat_eval, args.label_list_eval, args.subsampling)
