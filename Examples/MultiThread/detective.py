import cv2
import face_recognition

# Kameradan video akışını başlat
video_capture = cv2.VideoCapture(0)

# Yüzleri tespit etmek için bir liste oluştur
face_locations = []

# Video akışını sürekli olarak işle
while True:
    # Bir kare yakala
    ret, frame = video_capture.read()

    # Kareyi RGB formatına dönüştür
    rgb_frame = frame[:, :, ::-1]

    # Karede yüzleri tespit et
    face_locations = face_recognition.face_locations(rgb_frame)

    # Her bir yüzü çerçevele
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Kareyi görüntüle
    cv2.imshow("Video", frame)

    # "q" tuşuna basıldığında sonlandır
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kamerayı kapat
video_capture.release()

# Tüm pencereleri kapat
cv2.destroyAllWindows()
