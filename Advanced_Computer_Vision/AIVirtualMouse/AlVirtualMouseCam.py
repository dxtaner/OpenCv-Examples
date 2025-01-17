import cv2  # pip install opencv-contrib-python
import numpy as np
import mediapipe as mp  # pip install mediapipe
import pyautogui  # pip install PyAutoGUI

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Variables for smoothening
smoothening = 9
previous_location_x, previous_location_y = 0, 0
current_location_x, current_location_y = 0, 0
index_finger_y = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index finger tip
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(0, 255, 255))
                    index_finger_x = (screen_width / frame_width) * x
                    index_finger_y = (screen_height / frame_height) * y

                    # Smoothen the mouse movement
                    current_location_x = previous_location_x + (index_finger_x - previous_location_x) / smoothening
                    current_location_y = previous_location_y + (index_finger_y - previous_location_y) / smoothening
                    pyautogui.moveTo(current_location_x, current_location_y)
                    previous_location_x, previous_location_y = current_location_x, current_location_y

                if id == 4:  # Thumb tip
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(0, 255, 255))
                    thumb_x = (screen_width / frame_width) * x
                    thumb_y = (screen_height / frame_height) * y

                    # Check if the index finger and thumb are close to simulate a click
                    if abs(index_finger_y - thumb_y) < 70:
                        pyautogui.click()
                        pyautogui.sleep(1)

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
