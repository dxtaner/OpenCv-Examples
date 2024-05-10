import cv2

# Video dosyasını aç
cap = cv2.VideoCapture("Cars2.mp4")

# Arka plan çıkartıcı oluştur
subtractor = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=120, detectShadows=True)

# Daha büyük bir ekran boyutu belirle
new_width, new_height = 660, 440

while True:
    # Bir frame'i oku
    ret, frame = cap.read()

    # Eğer frame okunamazsa döngüyü sonlandır
    if not ret:
        break

    # Frame boyutunu belirlenen boyuta yeniden boyutlandır
    frame = cv2.resize(frame, (new_width, new_height))

    # Arka plan çıkartmayı uygula
    mask = subtractor.apply(frame)

    # Orijinal frame'i ve arka plan çıkartılmış mask'ı göster
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Mask Frame", mask)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Video dosyasını serbest bırak ve penceleri kapat
cap.release()
cv2.destroyAllWindows()
