import cv2
import numpy as np

kamera= cv2.VideoCapture(0)

while(1):
    ret, frame=kamera.read()

    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    dusuk_mavi=np.array([100,60,60])
    ust_mavi=np.array([140,255,255])    

    kenarlar = cv2.Canny(frame,100,150)
    #iap=cv2.Laplacian(frame,cv2.CV_64F)
    #sobelX=cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    #sobelY=cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)


    mask=cv2.inRange(hsv,dusuk_mavi,ust_mavi)
    son_resim=cv2.bitwise_and(frame,frame,mask=mask)


    cv2.imshow('orjinal',frame)
    cv2.imshow('cany', kenarlar)
    #cv2.imshow('lap', iap)
    #cv2.imshow('sobelX', sobelX)
    #cv2.imshow('sobelY', sobelY)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()