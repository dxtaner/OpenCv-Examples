import cv2

# Load the pre-trained Haar Cascade for cat faces
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")

# Use webcam for real-time detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cat faces
    cat_faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(75, 75))

    # Draw rectangles around detected cat faces
    for (x, y, w, h) in cat_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Cat Face Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
