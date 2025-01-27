import cv2
import time
import mediapipe as mp

def main():
    # Initialize video capture from the default camera
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Hands and drawing utilities
    mpHands = mp.solutions.hands
    mpDraw = mp.solutions.drawing_utils
    hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break

        # Convert the image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(imgRGB)

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Draw landmarks and connections on the image
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                # Draw a circle on the wrist (landmark id 0)
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 0:
                        cv2.circle(img, (cx, cy), 9, (255, 0, 0), cv2.FILLED)

        # Calculate Frames Per Second (FPS)
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        # Display FPS on the image in green with a plain font
        cv2.putText(img, f'FPS: {int(fps)}', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        # Display additional text in blue with a complex font
        cv2.putText(img, 'Hand Tracking', (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the image
        cv2.imshow("Hand Tracking", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
