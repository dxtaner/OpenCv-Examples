import numpy as np
import cv2

imgrgb=cv2.imread('ana_resim.jpg')
griton=cv2.cvtColor(imgrgb,cv2.COLOR_BGR2GRAY)

nesne=cv2.imread("template.jpg",0)

w,h=nesne.shape[::-1]
res=cv2.matchTemplate(griton,nesne,cv2.TM_CCOEFF_NORMED)
threshold=0.79

loc=np.where(res>threshold)

for n in zip(*loc[::-1]):
    cv2.rectangle(imgrgb,n,(n[0]+w,n[1]+h),(0,255,255),2)

cv2.imshow("bulunannesneler",imgrgb)
cv2.waitKey(0)
cv2.destroyAllWindows()