import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    detector = htm.handDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break

        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        # Display FPS in different font and color
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        # Display additional text in another font and color
        cv2.putText(img, 'Hand Tracking', (10, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the image
        cv2.imshow("Video", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
