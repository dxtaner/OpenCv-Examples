import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Modeli yükle ===
model = load_model("mask_model_binary.h5")

# === Sınıf etiketleri ===
class_names = ['Maskesiz', 'Maskeli']  # 0: maskesiz, 1: maskeli

# === Haar Cascade ile yüz tespiti ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Kamerayı başlat ===
cap = cv2.VideoCapture(0)
print("🎥 Kamera başlatıldı. Çıkmak için ESC'ye bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_normalized = face_resized / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)

        # === Tahmin ===
        prediction = model.predict(face_expanded)
        class_index = int(prediction[0][0] > 0.5)
        confidence = prediction[0][0] if class_index == 1 else 1 - prediction[0][0]
        label = f"{class_names[class_index]} ({confidence:.2f})"

        # === Çizim ===
        color = (0, 255, 0) if class_index == 1 else (0, 0, 255)  # Yeşil: maskeli, Kırmızı: maskesiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # === Göster ===
    cv2.imshow("Maske Tespiti (2 sinif)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC ile çık
        break

cap.release()
cv2.destroyAllWindows()
