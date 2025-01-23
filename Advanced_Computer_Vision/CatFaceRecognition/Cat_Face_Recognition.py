import cv2

# Load the pre-trained Haar Cascade for cat faces
cat_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

# Check if the cascade file is loaded
if cat_cascade.empty():
    raise FileNotFoundError("Haar Cascade file not found. Ensure OpenCV's haarcascade_frontalcatface.xml is available.")

# Read input image or video
# Use an image file (replace 'cat_image.jpg' with your image path)
image = cv2.imread("cats.png")
if image is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# Convert image to grayscale for better detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect cat faces
cat_faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(75, 75))

# Draw rectangles around detected cat faces
for (x, y, w, h) in cat_faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Display the results
cv2.imshow("Cat Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
