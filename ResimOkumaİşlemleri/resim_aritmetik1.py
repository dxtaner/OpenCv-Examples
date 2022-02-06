import cv2

img = cv2.imread("cicek.jpg")

print(img[80,80])
img[80,80]=[255,255,255]

bolge=img[30:120,100:200]
img[0:90,0:100]=bolge
cv2.rectangle(img,(100,30),(200,120),(0,255,0),2)
cv2.imshow("cicek",img)
cv2.imshow("cicek2",bolge)

cv2.waitKey(0)
cv2.destroyAllWindows()