import cv2

# Video dosyasını aç
vid = cv2.VideoCapture("cars2.mp4")

# Araba kaskadını yükle
car_cascade = cv2.CascadeClassifier("car.xml")

while True:
    # Bir frame'i oku
    ret, frame = vid.read()

    # Eğer frame okunamazsa döngüyü sonlandır
    if not ret:
        print("Video akışı alınamıyor. Çıkılıyor...")
        break

    # Frame boyutunu (640, 480) olarak değiştir
    frame = cv2.resize(frame, (640, 480))

    # Gri tonlamalı görüntüyü elde et
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Araba tespiti
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

    # Tespit edilen arabaları çerçeve içine al
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Sonucu ekranda göster
    cv2.imshow('Detected Cars', frame)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Video dosyasını serbest bırak ve penceleri kapat
vid.release()
cv2.destroyAllWindows()
