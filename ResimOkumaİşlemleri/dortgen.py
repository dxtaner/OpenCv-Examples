import cv2
import numpy as np

resim = np.zeros((400,400,3),dtype="uint8")

cv2.rectangle(resim,(10,10),(200,120),(0,255,255),3)
cv2.line(resim,(10,10),(200,120),(255,255,255),3)
cv2.line(resim,(10,230),(390,230),(230,120,255),3)
cv2.circle(resim,(50,280),25,(120,55,46),3)

cv2.putText(resim,"Taner",(180,320),cv2.FONT_HERSHEY_COMPLEX,2,(44,45,65),3,cv2.LINE_4)

cv2.imshow("siyah",resim)

cv2.waitKey(0)
cv2.destroyAllWindows()