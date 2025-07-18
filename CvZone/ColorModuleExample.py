import cv2
import cvzone  # Import the cvzone module for stackImages
from cvzone.ColorModule import ColorFinder  # Explicitly import the ColorFinder class

# Create an instance of the ColorFinder class with trackBar set to False
myColorFinder = ColorFinder(trackBar=False)

# Initialize the video capture using OpenCV
# Using the default camera (index 0). Adjust index if you have multiple cameras.
cap = cv2.VideoCapture(0)

# Set the dimensions of the camera feed to 640x480
cap.set(3, 640)
cap.set(4, 480)

# Custom color values for detecting orange
# 'hmin', 'smin', 'vmin' are the minimum values for Hue, Saturation, and Value
# 'hmax', 'smax', 'vmax' are the maximum values for Hue, Saturation, and Value
hsvVals = {'hmin': 4, 'smin': 0, 'vmin': 234, 'hmax': 116, 'smax': 255, 'vmax': 255}

# Main loop to continuously get frames from the camera
while True:
    # Read the current frame from the camera
    success, img = cap.read()

    # Check if the frame was successfully captured
    if not success:
        print("Error: Failed to capture frame.")
        break

    # Use the update method from the ColorFinder class to detect the color
    # It returns the masked color image and a binary mask
    imgOrange, mask = myColorFinder.update(img, hsvVals)

    # Stack the original image, the masked color image, and the binary mask
    imgStack = cvzone.stackImages([img, imgOrange, mask], 3, 1)

    # Show the stacked images
    cv2.imshow("Image Stack", imgStack)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()