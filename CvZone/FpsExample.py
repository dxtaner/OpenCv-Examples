import cv2
from cvzone.FPS import FPS  # Explicitly import the FPS class

# Initialize the FPS class with an average count of 30 frames for smoothing
fpsReader = FPS(avgCount=30)

# Initialize the webcam and set it to capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)  # Set the frames per second to 60

# Main loop to capture frames and display FPS
while True:
    # Read a frame from the webcam
    success, img = cap.read()

    # Check if the frame was successfully captured
    if not success:
        print("Error: Failed to capture frame.")
        break

    # Update the FPS counter and draw the FPS on the image
    # fpsReader.update returns the current FPS and the updated image
    fps, img = fpsReader.update(img, pos=(20, 50),
                                bgColor=(255, 255, 0), textColor=(255, 255, 255),
                                scale=3, thickness=3)
    print(fps)

    # Display the image with the FPS counter
    cv2.imshow("Image", img)

    # Wait for 1 ms to show this frame, then continue to the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()