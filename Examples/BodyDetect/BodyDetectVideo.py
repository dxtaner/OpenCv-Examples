import cv2

# 1. Video dosyasını aç
video_path = "body.mp4"
cap = cv2.VideoCapture(video_path)

# 2. Cascade dosyasını yükle
body_cascade = cv2.CascadeClassifier("fullbody.xml")

while True:
    # 3. Bir kareyi oku
    ret, frame = cap.read()

    # 4. Eğer kare okunamazsa döngüyü sonlandır
    if not ret:
        print("Video akışı alınamıyor. Çıkılıyor...")
        break

    # 5. Haar-like özellikleri kolay algılayabilmek için kareyi boz(gri) tonlara çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 6. Cascade dosyasını kullanarak her bir kare üzerindeki bedenleri bul
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=5)

    # 7. Bulunan bedenlerin koordinatlarını kare üzerine çiz
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 8. İşlenmiş kareyi ekranda göster
    cv2.imshow('Detected Bodies', frame)

    # 9. 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 10. Video dosyasını serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
