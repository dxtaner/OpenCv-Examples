import cv2
import numpy as np

img=cv2.imread('messi.jpg')

#img[:,:,1]=0
print(img[:,:,1])
alanmavi=img[:,:,0]
alanyesil=img[:,:,1]
alankirmizi=img[:,:,2]
img[:,:,1]=0
img[:,:,2]=0

cv2.imshow('alanmavi',alanmavi)
cv2.imshow('img',img)
#cv2.imshow('alanyesil',alanyesil)
#cv2.imshow('alankirmizi',alankirmizi)
cv2.waitKey(0)
cv2.destroyAllWindows()