import numpy as np
import cv2
from matplotlib import pyplot as plt

img=cv2.imread('tanneröz.jpg')
mask=np.zeros(img.shape[:2],np.uint8)

bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)

dortgen=(175,125,150,175)

cv2.grabCut(img,mask,dortgen,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2=np.where((mask==2) | (mask==0), 0, 1).astype("uint8")
img=img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()