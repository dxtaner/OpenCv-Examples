import cv2

# Video dosyasını aç
video_path = "eye_motion.mp4"
vid = cv2.VideoCapture(video_path)

while True:
    # Bir kareyi oku
    ret, frame = vid.read()

    # Eğer kare okunamazsa döngüyü sonlandır
    if not ret:
        print("Video akışı alınamıyor. Çıkılıyor...")
        break

    # ROI (Region of Interest) belirle ve gri tonlamalı hale getir
    roi = frame[80:210, 230:450]
    rows, cols, _ = roi.shape
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Threshold uygula
    _, threshold = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY_INV)

    # Contourları bul ve alanlarına göre sırala
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # En büyük contouru seç ve çizim yap
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2)
        break

    # Orijinal kare içine ROI'yi yerleştir
    frame[80:210, 230:450] = roi

    # İşlenmiş kareyi ekranda göster
    cv2.imshow("Eye Motion Detection", frame)
    cv2.imshow("Eye Motion Gray Detection", gray)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(80) & 0xFF == ord('q'):
        break

# Video dosyasını serbest bırak ve pencereyi kapat
vid.release()
cv2.destroyAllWindows()
