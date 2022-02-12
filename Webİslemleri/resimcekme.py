import cv2
from skimage import io

adresler = [
    "https://3.bp.blogspot.com/-yvrV6MUueGg/ToICp0YIDPI/AAAAAAAAADg/SYKg4dWpyC43AAfrDwBTR0VYmYT0QshEgCPcBGAYYCw/s1600/OpenCV_Logo.png",
    "https://www.facebook.com/images/fb_icon_325x325.png",
    "https://static9.depositphotos.com/1359043/1175/i/600/depositphotos_11752106-stock-photo-pink-butterfly-isolated-on-white.jpg"
]

for adres in adresler:
    print("%s yukleniyor"%(adres))
    resim = io.imread(adres)
    cv2.imshow('BGR format',resim)
    cv2.imshow('RGB format',cv2.cvtColor(resim,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
