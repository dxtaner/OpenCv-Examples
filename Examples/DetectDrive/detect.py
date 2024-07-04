import cv2
import time

# Yüz ve göz tanıma sınıflarını yükle
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Kamera yakalamasını başlat
cap = cv2.VideoCapture(0)

while True:
    # Kareyi yakala
    ret, frame = cap.read()

    # Kare üzerinde gri ölçek dönüşümü yap
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Yüzleri çerçeve içine al
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Yüz bölgesini gri ölçekte seç
        roi_gray = gray[y:y + h, x:x + w]

        # Yüz bölgesini renkli kare üzerinden seç
        roi_color = frame[y:y + h, x:x + w]

        # Gözleri tespit et
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Eğer göz yoksa ve belirli bir süre boyunca göz tespit edilmezse
        if not len(eyes):
            t = 0
            while not len(eyes):
                t += 1
                if t > 5:
                    # Sürücü uyuyor gibi görünüyor, metin ekleyin
                    cv2.putText(frame, "Surucu Uykuda!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    time.sleep(2)
                    break

        for (ex, ey, ew, eh) in eyes:
            # Gözleri çerçeve içine al
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)

    # Kareyi göster
    cv2.imshow('frame', frame)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak
cap.release()

# Pencereyi kapat
cv2.destroyAllWindows()
