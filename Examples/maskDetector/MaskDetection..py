import cv2
import sys

# Specify the path to the video file
video_path = "mask.mp4"

# Load the face cascade classifier
faceCascade = cv2.CascadeClassifier("haarcascade_mask.xml")

try:
    # Initialize video capture from the specified file
    video_capture = cv2.VideoCapture(video_path)

    # Check if video capture was successful
    if not video_capture.isOpened():
        raise RuntimeError("Error opening video file:", video_path)

    # Continuously capture and process frames from the video file
    while True:
        # Capture a frame from the video stream
        ret, frame = video_capture.read()

        # Check if frame capture was successful
        if not ret:
            print("End of video reached.")
            break

        try:
            # Convert the captured frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=9,
                minSize=(30, 30),
            )

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the processed frame with detected faces
            cv2.imshow('Video', frame)
        except Exception as e:
            print("Error processing frame:", e)

        # Check if 'q' key is pressed to exit
        if cv2.waitKey(33) == ord('q'):
            break

    # Release the video capture object
    video_capture.release()

except Exception as e:
    print("General error:", e)

# Close all open windows
cv2.destroyAllWindows()
