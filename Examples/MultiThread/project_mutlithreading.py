import numpy as np
import cv2
import threading
import datetime

def function_1():
    # Initialize video capture for camera 0
    video_capture_0 = cv2.VideoCapture(0)

    while True:
        # Capture frame from camera 0
        ret0, frame0 = video_capture_0.read()

        if ret0:
            # Display frame from camera 0
            cv2.imshow("Cam 1", frame0)

            # Check if 'q' key is pressed
            if cv2.waitKey(34) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object for camera 0
    video_capture_0.release()

    # Close all open windows
    cv2.destroyAllWindows()


def function_2():
    # Initialize video capture for IP camera
    # video_capture_1 = cv2.VideoCapture("/video")
    video_capture_1 = cv2.VideoCapture("car.mp4")

    while True:
        # Capture frame from IP camera
        ret1, frame1 = video_capture_1.read()

        if ret1:
            # Display frame from IP camera
            cv2.imshow("Cam 2", frame1)

            # Check if 'q' key is pressed
            if cv2.waitKey(34) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object for IP camera
    video_capture_1.release()

    # Close all open windows
    cv2.destroyAllWindows()

# Start threads for both camera feeds
t1 = threading.Thread(target=function_1)
t2 = threading.Thread(target=function_2)

t1.start()
t2.start()

# Wait for both threads to finish
t1.join()
t2.join()
