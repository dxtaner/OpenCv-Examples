import cv2

resim=cv2.imread('cicek.jpg')
cv2.imshow('cicek',resim)

cv2.rectangle(resim,(200,70),(320,180),(0,255,255),2)
cv2.imshow('dortgencevcere',resim)


cv2.waitKey(0)
cv2.destroyAllWindows()