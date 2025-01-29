import cv2
import numpy as np

# Read the input image
image = cv2.imread("indir.png")
if image is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove small noise with morphological opening
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Identify sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Identify sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Identify unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that the background is not 0
markers = markers + 1

# Mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply the Watershed algorithm
markers = cv2.watershed(image, markers)

# Mark boundaries in red on the original image
image[markers == -1] = [0, 0, 255]

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Binary Image", binary)
cv2.imshow("Watershed Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
