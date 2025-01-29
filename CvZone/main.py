import cv2

def check_cameras():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera found at index {index}")
            cap.release()
        else:
            print(f"No camera at index {index}")
            break
        index += 1

check_cameras()
