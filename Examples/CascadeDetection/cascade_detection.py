import cv2
import os
import requests

# === 1) Cascade dosyasını indir ===
def download_cascade(cascade_url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print("Cascade dosyası indiriliyor...")
        r = requests.get(cascade_url)
        with open(save_path, "wb") as f:
            f.write(r.content)
        print("Cascade indirildi.")
    else:
        print("Cascade zaten mevcut.")

cascade_url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"
cascade_path = "cascade_files/cascade.xml"
download_cascade(cascade_url, cascade_path)

# === 2) Kamera ve trackbar ayarları ===
def setup_camera_and_trackbar(width, height, window_name):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def empty(a): pass
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, width, height + 100)
    cv2.createTrackbar("Scale", window_name, 400, 1000, empty)
    cv2.createTrackbar("Neighbor", window_name, 4, 50, empty)

    return cap

frameWidth = 200
frameHeight = 360
windowName = "Sonuc"
cap = setup_camera_and_trackbar(frameWidth, frameHeight, windowName)

# === 3) Nesne tespiti ve gösterim ===
def detect_objects_and_show(cap, cascade_path, objectName, color, windowName):
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("Cascade dosyası yüklenemedi!")
        return

    while True:
        success, img = cap.read()
        if not success:
            print("Kamera görüntüsü alınamadı.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scaleVal = 1 + (cv2.getTrackbarPos("Scale", windowName) / 1000)
        neighbor = max(cv2.getTrackbarPos("Neighbor", windowName), 1)

        rects = cascade.detectMultiScale(gray, scaleVal, neighbor)

        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, objectName, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

        cv2.imshow(windowName, img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# === 4) Fonksiyonu başlat ===
objectName = "Fare"         # Ekranda yazacak isim
color = (255, 0, 0)         # Mavi renk kutu
detect_objects_and_show(cap, cascade_path, objectName, color, windowName)
