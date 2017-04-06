# endo-loc

This project seeks to enable the rough inside/outside-patient localization of an endoscope. At this stage the proposed feature vector is meant to be a HSV histogram of the incoming endoscope image stream.

This software is developed along the principles of test-driven development. This implies that the current abilities of the software can be deduced from the `test_...` methods in all "`tests_... .py` files in the tests package.

## Scripts

A typical workflow at the moment consist of running several of the scripts in the top directory. I recommend the following (assuming you labeled your videos with the [Aegisub subtitle editor](http://www.aegisub.org/)):

* Convert your video(s) to a feature XML file by calling `python convert_video_to_xml.py path_to_video`.
* Convert your labels file(s) (.ass) to a labels list file (.csv) by calling `python convert_labels_to_label_list.py path_to_labels_file`.
* Take two feature XMLs and their corresponding label list CSVs and call `python get_evaluation.py path_to_ftxml1 path_to_labellist1 path_to_ftxml2 path_to_labellist2`. The evaluation output will printed on the standard output.
