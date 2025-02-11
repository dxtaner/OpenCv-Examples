import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Try 0 or 1 instead of 2

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the frame width to 640 pixels
cap.set(3, 640)
# Set the frame height to 480 pixels
cap.set(4, 480)

# Initialize the SelfiSegmentation class
segmentor = SelfiSegmentation(model=0)

# Infinite loop to keep capturing frames from the webcam
while True:
    # Capture a single frame
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    # Use the SelfiSegmentation class to remove the background
    imgOut = segmentor.removeBG(img, imgBg=(0, 0, 255), cutThreshold=0.1)

    # Stack the original image and the image with background removed side by side
    imgStacked = cvzone.stackImages([img, imgOut], cols=2, scale=1)

    # Display the stacked images
    cv2.imshow("Image", imgStacked)

    # Check for 'q' key press to break the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()