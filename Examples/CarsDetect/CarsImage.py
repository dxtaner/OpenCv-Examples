import cv2

# Görüntüyü oku
img = cv2.imread("cars.jpg")

# Araba kaskadını yükle
car_cascade = cv2.CascadeClassifier("car.xml")

# Gri tonlamalı görüntüyü elde et
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Araba tespiti
cars = car_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=2)

# Tespit edilen arabaları çerçeve içine al
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

# Sonucu ekranda göster
cv2.imshow('Detected Cars', img)

# Bir tuşa basılmasını bekle
cv2.waitKey(0)
cv2.destroyAllWindows()
