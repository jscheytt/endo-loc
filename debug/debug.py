import cv2
from matplotlib import pyplot as plt

import feature_extraction.ft_extractor as fx

IMG_TITLE = "image"


class Debug:
    @classmethod
    def display_image(cls, image):
        """Display image in a new window. Window closes after pressing any key."""
        cv2.imshow(IMG_TITLE, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    def get_histograms_rgb(cls, image):
        """
        Retrieve RGB histograms of an image.
        :param image: OpenCV image object
        :return: List of histograms in order red, green, blue.
        """
        hist_blue = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_green = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_red = cv2.calcHist([image], [2], None, [256], [0, 256])
        return [hist_red, hist_green, hist_blue]

    @classmethod
    def plot_histograms(cls, hists, update_interval=0.0):
        """
        Plot histogram in new window in Red, Green and Blue.
        :param hists: List of 3 histogram channels
        :param update_interval: Time in seconds at which the window should be updated
        :return:
        """
        color = ('r', 'g', 'b')
        for idx, col in enumerate(color):
            plt.plot(hists[idx], color=col)
            plt.xlim([0, 256])
        plt.draw()
        plt.pause(update_interval)

    @classmethod
    def plot_histograms_live(cls, filename):
        """
        Plot histogram of a video frame by frame. Is not expected to work in real time.
        :param filename: Path to video file
        :return:
        """
        cap = cv2.VideoCapture(filename)

        while cap.isOpened():
            _, frame = cap.read()

            hists_frame = fx.FeatureExtractor.get_histograms_hsv(frame)

            # Clear current plot
            plt.clf()
            plt.cla()
            # Update at 25 fps speed if live
            Debug.plot_histograms(hists_frame, 0.04)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
