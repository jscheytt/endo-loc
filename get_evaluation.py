import argparse
import logging.config

import sample.sample as s


def main(feat_train, label_list_train, feat_eval, label_list_eval):
    # Load Logging settings
    logging.config.fileConfig("logging_config.ini")

    data_train, targets_train = s.get_data_and_targets(feat_train, label_list_train)
    svc = s.get_svclassifier(data_train, targets_train)
    data_eval, targets_eval = s.get_data_and_targets(feat_eval, label_list_eval)
    evaluation = s.get_evaluation(svc, data_eval, targets_eval)
    print(evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print evaluation metrics for learning an HSV classifier "
                                                 "on 2 sets of feature data + targets.")
    parser.add_argument("feat_train", help="The features file (.xml) for training the classifier")
    parser.add_argument("label_list_train", help="The label list file (.csv) for training the classifier")
    parser.add_argument("feat_eval", help="The features file (.xml) for evaluating the classifier")
    parser.add_argument("label_list_eval", help="The label list file (.csv) for evaluating the classifier")
    args = parser.parse_args()
    main(args.feat_train, args.label_list_train, args.feat_eval, args.label_list_eval)
