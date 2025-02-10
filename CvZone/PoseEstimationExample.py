from cvzone.PoseModule import PoseDetector
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 or 1 for the webcam index

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize the PoseDetector class
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)

# Define a beautiful color palette
COLORS = {
    "background": (40, 40, 40),  # Dark gray for background
    "landmarks": (0, 255, 255),  # Cyan for landmarks
    "lines": (255, 0, 150),      # Pink for connecting lines
    "bbox": (0, 150, 255),       # Orange for bounding box
    "center": (0, 255, 0),       # Green for center point
    "distance": (255, 0, 0),     # Blue for distance line
    "angle": (0, 0, 255),        # Red for angle arc
}

# Loop to continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    # Find the human pose in the frame
    img = detector.findPose(img)

    # Find the landmarks, bounding box, and center of the body in the frame
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

    # Check if any body landmarks are detected
    if lmList:
        # Get the center of the bounding box around the body
        center = bboxInfo["center"]

        # Draw a circle at the center of the bounding box
        cv2.circle(img, center, 8, COLORS["center"], cv2.FILLED)
        cv2.circle(img, center, 12, COLORS["center"], 2)

        # Draw the bounding box
        x, y, w, h = bboxInfo["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), COLORS["bbox"], 2)

        # Calculate the distance between landmarks 11 and 15 and draw it on the image
        length, img, info = detector.findDistance(lmList[11][0:2],
                                                  lmList[15][0:2],
                                                  img=img,
                                                  color=COLORS["distance"],
                                                  scale=10)

        # Calculate the angle between landmarks 11, 13, and 15 and draw it on the image
        angle, img = detector.findAngle(lmList[11][0:2],
                                        lmList[13][0:2],
                                        lmList[15][0:2],
                                        img=img,
                                        color=COLORS["angle"],
                                        scale=10)

        # Check if the angle is close to 50 degrees with an offset of 10
        isCloseAngle50 = detector.angleCheck(myAngle=angle,
                                             targetAngle=50,
                                             offset=5)

        # Print the result of the angle check
        print(f"Angle Close to 50°: {isCloseAngle50}")

    # Add a dark background overlay for better visibility
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), COLORS["background"], -1)
    img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)

    # Display the frame in a window
    cv2.imshow("Pose Estimation", img)

    # Wait for 1 millisecond between each frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()