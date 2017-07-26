import argparse

import debug.debug as dbg
import helper.helper as hlp
import prep.preprocessor as pre
import sample.sample as s


def main(dir_train, C, gamma, number_partitions, do_subsampling, write_labels):
    hlp.setup_logging()

    # Files as folds?
    if number_partitions is None or number_partitions == 0:  # Yes
        do_concat = False
        partitions_from_files = True
        early_subsampling = False
        late_subsampling = True
    else:  # No
        do_concat = True
        partitions_from_files = False
        early_subsampling = True
        late_subsampling = False

    if not do_subsampling:
        early_subsampling = late_subsampling = False

    X, y = pre.get_multiple_data_and_targets(dir_filepath=dir_train,do_subsampling=early_subsampling,
                                             do_concat=do_concat)
    clf = s.get_svclassifier(C=C, gamma=gamma)
    evaluation, y_pred = s.get_crossval_evaluation(X, y, n_folds=number_partitions, print_scores=True, clf=clf,
                                                   files_as_folds=partitions_from_files,
                                                   do_subsampling=late_subsampling)

    if write_labels:
        dbg.write_list_to_dir(dir_train, y_pred, "y_pred.txt")
        if do_concat:
            dbg.write_list_to_dir(dir_train, y, "y_true.txt")

    hlp.log(evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print evaluation metrics for cross validating an HSV classifier.")
    parser.add_argument("dir_train",
                        help="Directory containing all feature XMLs and label CSVs for cross validating the "
                             "classifier. CSVs need to have the same file name as their corresponding XML.")
    parser.add_argument("-c", "--C_value", help="Omit the grid search and directly specify a C value.", type=float)
    parser.add_argument("-g", "--gamma_value", help="Omit the grid search and directly specify a gamma value.",
                        type=float)
    parser.add_argument("-p", "--number_partitions",
                        help="Set the number of partitions for cross validation. If omitted, take each file "
                             "as a partition.", type=int)
    parser.add_argument("-s", "--subsampling", help="Subsample majority class", action="store_true")
    parser.add_argument("-wl", "--write_labels",
                        help="Write both true and predicted labels of the eval file(s) to TXT files.",
                        action="store_true")
    args = parser.parse_args()
    main(args.dir_train, args.C_value, args.gamma_value, args.number_partitions, args.subsampling, args.write_labels)
