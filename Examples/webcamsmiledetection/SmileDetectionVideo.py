import cv2

# Video dosyasını aç
vid = cv2.VideoCapture("smile.mp4")

# Gülümseme ve yüz kaskadlarını yükle
smile_cascade = cv2.CascadeClassifier("smile.xml")
face_cascade = cv2.CascadeClassifier("frontalface.xml")

# Video akışı devam ettiği sürece
while True:
    # Bir frame'i oku
    ret, frame = vid.read()

    # Eğer frame okunamazsa döngüyü sonlandır
    if not ret:
        print("Video akışı alınamıyor. Çıkılıyor...")
        break

    # Frame boyutunu (720, 480) olarak değiştir
    frame = cv2.resize(frame, (720, 480))

    # Gri tonlamalı görüntüyü elde et
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Yüzü çerçeve içine al
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Yüz bölgesini gri tonlamalı görüntüden seç
        roi_gray = gray[y:y + h, x:x + w]
        roi_img = frame[y:y + h, x:x + w]

        # Gülümseme tespiti sadece yüz bölgesinde yapılır
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5)

        for (ex, ey, ew, eh) in smiles:
            # Gülümsemeyi çerçeve içine al
            cv2.rectangle(roi_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Sonuçları ekranda göster
    cv2.imshow('Video', frame)
    cv2.imshow('Grayscale Video', gray)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Video dosyasını serbest bırak ve penceleri kapat
vid.release()
cv2.destroyAllWindows()
