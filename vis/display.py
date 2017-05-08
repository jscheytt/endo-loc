import tkinter

import cv2

WINDOW_TITLE = 'video'
TKINTER_ROOT = tkinter.Tk()


def display_video(filename):
    """
    Display a video in a window. A wrapper for process_video with show_frame.
    :param filename: Path to video file
    :return: 
    """
    process_video(filename, show_frame)


def process_video(filename, action):
    """
    Process a video frame by frame, executing the action on every frame.
    :param filename: Path to the video file
    :param action: Function to be called, must have 1 parameter (frame)
    :return: 
    """
    cap = cv2.VideoCapture()
    if not cap.isOpened():
        cap.open(filename)

    while cap.isOpened():
        ret, frame = cap.read()

        action(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def show_frame(frame, fullscreen=False):
    """
    Action function for process_video: Simply display the frame.
    :param frame: video frame to be displayed
    :param fullscreen: Show frame in fullscreen window
    :return: 
    """
    if fullscreen:
        cv2.namedWindow(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WINDOW_TITLE, frame)


def load_image(filepath):
    """
    Read an image from disk.
    :param filepath: 
    :return: 
    """
    return cv2.imread(filepath, cv2.IMREAD_COLOR)


def process_image(img, action):
    """
    Process an image and display it in a window.
    Window closes after pressing any key.
    :param img: 
    :param action: name of the function to be executed upon img
    :return: 
    """
    action(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_screen_dims():
    """
    Get width and height of the current screen.
    :return: tuple(width, height)
    """
    # TODO may not work with multiple monitors, see
    # http://stackoverflow.com/questions/3129322/how-do-i-get-monitor-resolution-in-python/31171430#31171430
    # alternative solutions:
    # https://www.blog.pythonlibrary.org/2015/08/18/getting-your-screen-resolution-with-python/
    width = TKINTER_ROOT.winfo_screenwidth()
    height = TKINTER_ROOT.winfo_screenheight()
    return width, height
