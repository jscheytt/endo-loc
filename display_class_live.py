import argparse

import helper.helper as hlp
import sample.sample as s
import vis.classify_live as cllv
import vis.display as dsp


def main(video, clf, skip_frames, no_h_channel=False, no_s_channel=False, no_v_channel=False):
    hlp.setup_logging()

    # Cast parameter video to int if possible
    try:
        video = int(video)
    except ValueError:
        pass

    h_c = not no_h_channel
    s_c = not no_s_channel
    v_c = not no_v_channel

    cllv.CLF = s.read_classifier(clf)

    def predictor(frame, skip_frames):
        return cllv.display_predict_on_frame(frame, skip_frames=skip_frames, h_c=h_c, s_c=s_c, v_c=v_c)

    dsp.process_video(video, predictor, skip_frames=skip_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display live video from a camera and classify each frame. Quit with 'Q' key.")
    parser.add_argument("video", help="Video file OR camera index OR camera IP address")
    parser.add_argument("classifier", help="Dump (.pkl) from scikit-learn containing the classifier")
    parser.add_argument("-sf", "--skip_frames", help="Skip every n frames from classifying (not from display)",
                        default=0)
    parser.add_argument("-noh", "--no_h_channel", help="Classifier was learned without H channel", action="store_true")
    parser.add_argument("-nos", "--no_s_channel", help="Classifier was learned without S channel", action="store_true")
    parser.add_argument("-nov", "--no_v_channel", help="Classifier was learned without V channel", action="store_true")
    args = parser.parse_args()
    main(args.video, args.classifier, args.skip_frames, args.no_h_channel, args.no_s_channel, args.no_v_channel)
