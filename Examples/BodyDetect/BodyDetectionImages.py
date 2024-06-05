import cv2

# 2. Kullanacağımız resmi çalışmamıza dahil edelim.
img = cv2.imread("body.jpg")

# 3. Kullanacağımız cascade dosyalarını çalışmamıza dahil edelim.
body_cascade = cv2.CascadeClassifier("fullbody.xml")

# 4. Haar-like özellikleri kolay algılayabilmek için resmi boz(gri) tonlara çevirelim.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 5. Cascade dosyamızı kullanarak her bir kare üzerindeki bedenleri bulalım.
bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=8)

# 6. Bulunan bedenlerin koordinatlarını ekrana çizelim.
for (x, y, w, h) in bodies:
    # Dikdörtgen içerisine alma işlemi
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 7. İşlenmiş kareleri görelim.
cv2.imshow('Detected Bodies', img)
cv2.imshow('Detected Bodies Gray', gray)

# 8. Bir tuşa basılmasını bekleyelim.
cv2.waitKey(0)

# 9. Pencereyi kapatma işlemi
cv2.destroyAllWindows()
