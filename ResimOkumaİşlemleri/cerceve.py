import cv2
import numpy as np
from matplotlib import pyplot as plt

renk=[244,150,90]

img=cv2.imread('opencv.JPG')

replicate=cv2.copyMakeBorder(img,9,9,9,9,cv2.BORDER_REPLICATE)
reflect=cv2.copyMakeBorder(img,9,9,9,9,cv2.BORDER_REFLECT)
reflect101=cv2.copyMakeBorder(img,9,9,9,9,cv2.BORDER_REFLECT101)
wrap=cv2.copyMakeBorder(img,9,9,9,9,cv2.BORDER_WRAP)
constant=cv2.copyMakeBorder(img,9,9,9,9,cv2.BORDER_CONSTANT,value=renk)

plt.subplot(231),plt.imshow(img,'gray'),plt.title('orjinal')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('replicate')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('reflect')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('reflect101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('wrap')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('constant')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
