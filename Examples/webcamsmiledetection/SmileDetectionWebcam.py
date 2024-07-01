import cv2

# Gülümseme ve yüz kaskadlarını yükle
# Gülümseme ve yüz kaskadlarını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Kaskad dosyalarını başarıyla yüklediğimizi kontrol et
if face_cascade.empty() or smile_cascade.empty():
    print('Kaskad dosyalarını yükleyemedi. Dosya yollarını kontrol edin.')
    exit()

# Kamera akışını başlat
vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()

    if not ret:
        print("Kamera akışı alınamıyor. Çıkılıyor...")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Gülümseme tespiti sadece yüz bölgesinde yapılır
        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.8, minNeighbors=20)

        for (ex, ey, ew, eh) in smiles:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)

    # Orijinal ve gri görüntüleri ekranda göster
    cv2.imshow('Original Video', frame)
    cv2.imshow('Grayscale Video', gray)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera akışını serbest bırak ve penceleri kapat
vid.release()
cv2.destroyAllWindows()
