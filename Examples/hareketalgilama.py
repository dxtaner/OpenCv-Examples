import cv2
from datetime import datetime

def farkimaj(t0,t1,t2):
    fark1=cv2.absdiff(t2,t1)
    fark2=cv2.absdiff(t1,t0)
    return cv2.bitwise_and(fark1,fark2)

esikdeger=250000
kamera=cv2.VideoCapture(0)

pencereismi="hareketalgilayici"
cv2.namedWindow(pencereismi)

t_eksi=cv2.cvtColor(kamera.read()[1],cv2.COLOR_BGR2GRAY)
t=cv2.cvtColor(kamera.read()[1],cv2.COLOR_BGR2GRAY)
t_arti=cv2.cvtColor(kamera.read()[1],cv2.COLOR_BGR2GRAY)

zamankontrol=datetime.now().strftime('%Ss')

while True:
    cv2.imshow(pencereismi,kamera.read()[1])
    if cv2.countNonZero(farkimaj(t_eksi,t,t_arti))>esikdeger and zamankontrol != datetime.now().strftime('%Ss'):
        farkresim=kamera.read()[1]
        cv2.imwrite(datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss%f")+".jpg",farkresim)
    zamankontrol=datetime.now().strftime('%Ss')
    t_eksi=t
    t=t_arti
    t_arti=cv2.cvtColor(kamera.read()[1],cv2.COLOR_BGR2GRAY)
    key=cv2.waitKey(10)
    if key==27:
        cv2.destroyWindows(pencereismi)
        break