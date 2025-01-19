import cv2
import mediapipe as mp
import time
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None  # Initialize results as None

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        landmark_list = []
        if self.results and self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[hand_no]
            for id, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (0, 150, 255), cv2.FILLED)  # Bright orange for landmarks
        return landmark_list


class VolumeController:
    def __init__(self):
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        self.volume = self.interface.QueryInterface(IAudioEndpointVolume)
        self.vol_range = self.volume.GetVolumeRange()
        self.min_vol = self.vol_range[0]
        self.max_vol = self.vol_range[1]

    def set_volume(self, volume_level):
        self.volume.SetMasterVolumeLevel(volume_level, None)


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    volume_controller = VolumeController()
    prev_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.find_hands(img)
        landmark_list = detector.find_position(img, draw=False)

        # Initialize variables
        vol = 0
        vol_bar = 400
        vol_per = 0

        if landmark_list:
            x1, y1 = landmark_list[4][1], landmark_list[4][2]
            x2, y2 = landmark_list[8][1], landmark_list[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 8, (0, 255, 0), cv2.FILLED)  # Bright green for index finger
            cv2.circle(img, (x2, y2), 8, (0, 255, 0), cv2.FILLED)  # Bright green for thumb
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)     # Bright green for line
            cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)  # Magenta for center point

            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [30, 200], [volume_controller.min_vol, volume_controller.max_vol])
            vol_bar = np.interp(length, [30, 200], [400, 150])
            vol_per = np.interp(length, [30, 200], [0, 100])

            volume_controller.set_volume(vol)
            if length < 50:
                cv2.circle(img, (cx, cy), 8, (0, 255, 255), cv2.FILLED)  # Cyan when close

        # Draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 255), 3)  # White border for volume bar
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 128, 0), cv2.FILLED)  # Dark green fill
        cv2.putText(img, f'{int(vol_per)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 2)  # Cyan for text

        # Draw FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 255), 2)  # Magenta for FPS

        cv2.imshow("Video Gesture Control", img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
