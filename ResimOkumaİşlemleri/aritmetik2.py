import cv2
import numpy as np

#x = np.uint8([250])
#y = np.uint8([10])

#print(x+y)
#print(cv2.add(x,y))

img1 = cv2.imread("cicek.png")
img2 = cv2.imread("messi.jpg")

toplam = cv2.addWeighted(img1,0.4,img2,0.6,0)
cv2.imshow("toplam",toplam)

cv2.waitKey(0)
cv2.destroyAllWindows()

