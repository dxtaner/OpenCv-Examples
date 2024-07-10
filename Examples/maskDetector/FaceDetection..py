import cv2
import sys

# Define the cascade classifier path
cascPath = "haarcascade-frontalface-default.xml"

# Load the face cascade classifier
faceCascade = cv2.CascadeClassifier(cascPath)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Continuously capture and process frames
while True:
    # Capture a frame from the video stream
    ret, frame = video_capture.read()

    # Check if frame capture was successful
    if not ret:
        print("Error: Unable to capture frame")
        break

    # Convert the captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the processed frame with detected faces
    cv2.imshow('Video', frame)

    # Check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video_capture.release()

# Close all open windows
cv2.destroyAllWindows()
