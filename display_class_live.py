import argparse

import helper.helper as hlp
import sample.sample as s
import vis.classify_live as cllv
import vis.display as dsp


def main(video, clf, skip_frames):
    hlp.setup_logging()

    # Cast parameter video to int if possible
    try:
        video = int(video)
    except ValueError:
        pass

    cllv.CLF = s.read_classifier(clf)
    dsp.process_video(video, cllv.display_predict_on_frame, skip_frames=skip_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display live video from a camera and classify each frame. Quit with 'Q' key.")
    parser.add_argument("video", help="Video file OR camera index OR camera IP address")
    parser.add_argument("classifier", help="Dump (.pkl) from scikit-learn containing the classifier")
    parser.add_argument("-sf", "--skip_frames", help="Skip every n frames from classifying (not from display)",
                        default=0)
    args = parser.parse_args()
    main(args.video, args.classifier, args.skip_frames)
