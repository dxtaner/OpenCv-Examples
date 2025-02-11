import cv2
from cvzone.Utils import rotateImage

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Start with the first webcam (index 0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Start the loop to continuously get frames from the webcam
while True:
    # Read a frame from the webcam
    success, img = cap.read()  # 'success' will be True if the frame is read successfully, 'img' will contain the frame

    if not success:
        print("Failed to capture image. Exiting...")
        break

    # Rotate the image by 60 degrees without keeping the size
    imgRotated60 = rotateImage(img, 60, scale=1, keepSize=False)

    # Rotate the image by 60 degrees while keeping the size
    imgRotated60KeepSize = rotateImage(img, 90, scale=1, keepSize=True)

    # Display the rotated images
    cv2.imshow("Original Image", img)
    cv2.imshow("Rotated 60 Degrees (No Keep Size)", imgRotated60)
    cv2.imshow("Rotated 90 Degrees (Keep Size)", imgRotated60KeepSize)

    # Wait for 1 millisecond and break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
