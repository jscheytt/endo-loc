# endo-loc

This project seeks to enable the rough inside/outside-patient localization of an endoscope. At this stage the proposed feature vector is meant to be a HSV histogram of the incoming endoscope image stream.

This software is developed along the principles of test-driven development. This implies that the current abilities of the software can be deduced from the `test_...` methods in all `tests_... .py` files in the tests package.

## Scripts

A typical workflow at the moment consist of running several of the scripts in the top directory. I recommend the following (assuming you labeled your videos with the [Aegisub subtitle editor](http://www.aegisub.org/)):

* Convert your video(s) to a feature XML file by calling `python convert_video_to_xml.py [path_to_video]` for each video.
* Convert your labels file(s) (.ass) to a labels list file (.csv) by calling `python convert_labels_to_label_list.py [path_to_labels_file] [path_to_corresponding_feature_file]`.
* Take a directory of feature XMLs and their corresponding label list CSVs and call `python get_evaluation.py [--dir_eval] [--subsampling] [dir_train or path_to_classifier]`. If `[dir_eval]` is omitted, cross validation is performed. The evaluation output of the learned classifier will be printed to standard output.
* Learn and export a classifier from a directory of feature XMLs and their corresponding label list CSVs with `python get_classifier.py [dir_train] [dir_eval] [path_to_classifier] [--C_value] [--gamma_value]`. Grid search can be skipped by supplying C and gamma values.
* Visualize your classifier by calling `python display_class_live.py [path_to_video or camera_index or camera_IP_address] [path_to_classifier] [--skip_frames]`.
