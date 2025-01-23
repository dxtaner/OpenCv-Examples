# Import necessary libraries
import cv2
import numpy as np

# Load the image
img = cv2.imread("people.jpg", 0)
if img is None:
    raise FileNotFoundError("'people.jpg' not found. Please check the file path.")

# Display the original image
cv2.imshow("Original Image", img)

# Perform edge detection using the Canny algorithm
edges = cv2.Canny(image=img, threshold1=200, threshold2=255)
cv2.imshow("Edge Detection", edges)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise FileNotFoundError("Haar Cascade file not found. Ensure access to OpenCV's haarcascades folder.")

# Perform face detection
face_rects = face_cascade.detectMultiScale(img, scaleFactor=1.03, minNeighbors=5, minSize=(30, 30))

# Convert the grayscale image to BGR for visualization
img_faces = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Draw rectangles around detected faces
for (x, y, w, h) in face_rects:
    cv2.rectangle(img_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("Face Detection", img_faces)

# Initialize the HOG descriptor for person detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Perform person detection
(rects, weights) = hog.detectMultiScale(img, padding=(8, 8), scale=1.05)

# Draw rectangles around detected people
for (x, y, w, h) in rects:
    cv2.rectangle(img_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("Person Detection", img_faces)

# Wait for a key press and close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()
