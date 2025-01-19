import cv2
import mediapipe as mp
import time
import os
import numpy as np

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        """Initialize the MediaPipe Hands module."""
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.track_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        """Detect hands in the image and optionally draw landmarks."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        """Get landmark positions of the detected hand."""
        lmlist = []
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[hand_no]
            h, w, _ = img.shape

            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        return lmlist

def load_overlay_images(folderpath):
    """Load overlay images from the specified folder."""
    overlay_images = {}
    for i in range(0, 6):  # Load images from 0 to 5
        image_path = os.path.join(folderpath, f'{i}.png')
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read with alpha channel
        if image is not None:
            overlay_images[i] = image
        else:
            print(f"Warning: {image_path} could not be loaded.")
    return overlay_images

def apply_overlay(img, overlay_img):
    """Apply overlay image with transparency effect."""
    # Ensure overlay image fits within the video frame
    img_h, img_w, _ = img.shape
    h, w, _ = overlay_img.shape
    if h <= img_h and w <= img_w:
        # Resize overlay to fit within the frame if necessary
        overlay_img = cv2.resize(overlay_img, (w, h), interpolation=cv2.INTER_AREA)
        alpha_overlay = overlay_img[:, :, 3] / 255.0
        alpha_img = 1.0 - alpha_overlay

        # Apply overlay with transparency
        for c in range(0, 3):
            img[0:h, 0:w, c] = (alpha_overlay * overlay_img[:, :, c] +
                                alpha_img * img[0:h, 0:w, c])

def main():
    folderpath = "FingerImages"
    overlay_images = load_overlay_images(folderpath)

    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    p_time = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        img = detector.find_hands(img)
        lmlist = detector.find_position(img, draw=False)

        # Detect fingers
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []

        if lmlist:
            # Thumb
            fingers.append(1 if lmlist[tip_ids[0]][1] > lmlist[tip_ids[0] - 1][1] else 0)

            # Other fingers
            for id in range(1, 5):
                fingers.append(1 if lmlist[tip_ids[id]][2] < lmlist[tip_ids[id] - 2][2] else 0)

        finger_count = sum(fingers)
        overlay_img = overlay_images.get(finger_count, None)

        if overlay_img is not None:
            apply_overlay(img, overlay_img)

        # Display the video feed with overlay image
        cv2.imshow("Hand Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
