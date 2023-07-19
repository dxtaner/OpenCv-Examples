import cv2
import numpy as np
import matplotlib.pyplot as plt

aranacakresim = cv2.imread("kucuk_resim.JPG",0)
buyukimg = cv2.imread("buyuk_resim.JPG",0)

orb=cv2.ORB_create()
an1,hedef1=orb.detectAndCompute(aranacakresim,None)
an2,hedef2=orb.detectAndCompute(buyukimg,None)
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

eslesmeler=bf.match(hedef1,hedef2)
eslesmeler=sorted(eslesmeler,key=lambda x:x.distance)
sonresim=cv2.drawMatches(aranacakresim,an1,buyukimg,an2,eslesmeler[:10],None,flags=2)

plt.imshow(sonresim)
plt.show()