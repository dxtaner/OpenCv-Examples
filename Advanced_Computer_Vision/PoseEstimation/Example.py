import cv2
import time
import PoseModule as pm


def main():
    cap = cv2.VideoCapture('PoseVideos/video4.mp4')

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    pTime = 0
    detector = pm.poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("End of video or failed to read frame.")
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Video", img)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
