import cv2
import cvzone

# Initialize camera capture
cap = cv2.VideoCapture(0)  # Replace 0 with the correct camera index

# Load the PNG image with transparency
img_png = cv2.imread("cvzoneLogo.png", cv2.IMREAD_UNCHANGED)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open the camera. Please check the camera index.")
    exit()

# Check if the PNG image is loaded successfully
if img_png is None:
    print("Failed to load the PNG image. Ensure 'cvzoneLogo.png' exists in the correct path.")
    exit()

while True:
    # Capture a frame from the camera
    success, img = cap.read()

    if not success:
        print("Failed to capture frame from the camera.")
        break

    # Overlay the PNG image at multiple positions
    img_overlay = cvzone.overlayPNG(img, img_png, pos=[10, 50])
    img_overlay = cvzone.overlayPNG(img_overlay, img_png, pos=[200, 200])
    img_overlay = cvzone.overlayPNG(img_overlay, img_png, pos=[500, 400])

    # Display the result
    cv2.imshow("Image with Overlay", img_overlay)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
