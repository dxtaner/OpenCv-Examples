import cv2
import cvzone

# Download and load the normal image
img_normal = cvzone.downloadImageFromUrl(
    url="https://github.com/cvzone/cvzone/blob/master/Results/shapes.png?raw=true"
)

# Download and load the transparent image with transparency preserved
img_png = cvzone.downloadImageFromUrl(
    url="https://github.com/cvzone/cvzone/blob/master/Results/cvzoneLogo.png?raw=true",
    keepTransparency=True
)

# Resize the transparent image
img_png = cv2.resize(img_png, (0, 0), None, 3, 3)

# Display the images
cv2.imshow("Normal Image", img_normal)
cv2.imshow("Transparent Image", img_png)
cv2.waitKey(0)
cv2.destroyAllWindows()
