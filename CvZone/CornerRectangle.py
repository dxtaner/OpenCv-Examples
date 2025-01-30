import cv2
import cvzone

cap = cv2.VideoCapture(0)  # Update the index based on available cameras

if not cap.isOpened():
    print("Failed to open the camera. Please check the camera index.")
    exit()

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture frame.")
        break

    cvzone.cornerRect(img, (200, 200, 300, 200))

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
