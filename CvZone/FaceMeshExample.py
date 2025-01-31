from cvzone.FaceMeshModule import FaceMeshDetector
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Try 0 or 1 instead of 2

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize FaceMeshDetector object
detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

# Start the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    # Find face mesh in the image
    img, faces = detector.findFaceMesh(img, draw=True)

    # Check if any faces are detected
    if faces:
        # Loop through each detected face
        for face in faces:
            # Get specific points for the eye
            leftEyeUpPoint = face[159]
            leftEyeDownPoint = face[23]
            # Calculate the vertical distance between the eye points
            leftEyeVerticalDistance, info = detector.findDistance(leftEyeUpPoint, leftEyeDownPoint)
            # Print the vertical distance for debugging or information
            print(leftEyeVerticalDistance)

    # Display the image in a window named 'Image'
    cv2.imshow("Image", img)

    # Wait for 1 millisecond to check for any user input, keeping the window open
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()