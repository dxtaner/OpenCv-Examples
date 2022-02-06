import cv2
import numpy as np

img1 = cv2.imread("messi.jpg")
img2 = cv2.imread("opencv.JPG")

satir,sutun,kanal = img2.shape
roi = img1[0:satir,0:sutun]

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
cv2.imshow("img1gray",img2gray)
ret, mask = cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow("mask",mask)
cv2.imshow("mask_inv",mask_inv)


img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
cv2.imshow("img1_bg",img1_bg)

img2_fg = cv2.bitwise_and(img2,img2,mask=mask)
cv2.imshow("img2_fg",img2_fg)

sonresim = cv2.add(img1_bg,img2_fg)
img1[0:satir,0:sutun]=sonresim

cv2.imshow("sonresim",img1)

cv2.waitKey(0)
cv2.destroyAllWindows()