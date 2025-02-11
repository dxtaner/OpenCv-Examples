import cv2
import cvzone

cap = cv2.VideoCapture(0)  # Replace with the correct index

if not cap.isOpened():
    print("Failed to open the camera. Please check the camera index.")
    exit()

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture frame.")
        break

    cv2.rectangle(img, (80, 40), (300, 200), (0, 255, 255), cv2.FILLED)
    cv2.putText(img, "CVZone", (100, 150), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)

    cvzone.putTextRect(img, "CVZone", (200, 300), border=5, offset=20)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
