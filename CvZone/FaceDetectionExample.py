import cv2
from cvzone.FaceDetectionModule import FaceDetector

# Initialize the webcam
# Use index 0 for the default camera, or adjust if you have multiple cameras
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize the FaceDetector object
# minDetectionCon: Minimum detection confidence threshold
# modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Run the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    # success: Boolean, whether the frame was successfully grabbed
    # img: the captured frame
    success, img = cap.read()

    # Check if the frame was successfully captured
    if not success or img is None:
        print("Error: Failed to capture frame.")
        break

    # Detect faces in the image
    # img: Updated image
    # bboxs: List of bounding boxes around detected faces
    img, bboxs = detector.findFaces(img, draw=False)

    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'

            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

            # ---- Draw Data  ---- #
            cv2.circle(img, center, 5, (255, 255, 255), cv2.FILLED)
            cv2.putText(img, f'{score}%', (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 250, 255), 2)

    # Display the image in a window named 'Image'
    cv2.imshow("Image", img)

    # Wait for 1 millisecond, and keep the window open
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()