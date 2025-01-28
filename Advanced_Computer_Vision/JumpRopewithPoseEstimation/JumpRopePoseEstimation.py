import cv2
import mediapipe as mp
import time
import math
import numpy as np


def findAngle(img, p1, p2, p3, lmList, draw=True):
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]

    # Açı hesaplama
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (0, 128, 255), 3)
        cv2.line(img, (x2, y2), (x3, y3), (0, 128, 255), 3)
        cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x3, y3), 10, (255, 255, 0), cv2.FILLED)

        cv2.circle(img, (x1, y1), 15, (255, 255, 0))
        cv2.circle(img, (x2, y2), 15, (255, 255, 0))
        cv2.circle(img, (x3, y3), 15, (255, 255, 0))

        cv2.putText(img, str(int(angle)), (x2 - 40, y2 + 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 155, 255), 2)
    return angle


# Mediapipe Pose sınıfını oluştur
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Mediapipe drawing utils nesnesini oluştur
mp_draw = mp.solutions.drawing_utils

# Video akışını aç
cap = cv2.VideoCapture("video3.mp4")  # Eğer bir video dosyası kullanılıyorsa, dosya adını belirtin

pTime = 0
cTime = 0

while True:
    # Video akışından bir kare oku
    success, img = cap.read()
    # img = cv2.resize(img, (1200, 900))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Görüntüyü pose algoritmasıyla işle
    results = pose.process(imgRGB)
    lmList = []

    if results.pose_landmarks:
        # Pose landmarklarını çiz
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Landmark koordinatlarını listeye ekle
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

            # Örneğin, belirli bir landmark üzerine işaretçi koyabilirsiniz (örneğin, 13 numaralı landmark)
            if id == 13:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Mor renk

        # Açıları hesapla ve çiz
        if lmList:
            angle = findAngle(img, 11, 13, 15, lmList)
            angle2 = findAngle(img, 12, 14, 16, lmList)

            # Açıları belirli bir aralıkta normalize et
            per = np.interp(angle, (90, 170), (0, 100))
            per2 = np.interp(angle2, (190, 270), (0, 100))

            # Belirlenen açı aralığına göre kontrolleri yap ve uygun mesajı göster
            if (0 < per < 100) and (0 < per2 < 100) and (abs(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y -
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y) < 0.03) and (abs(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y - results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_KNEE].y) < 0.03) and (abs(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x - results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_KNEE].x) < 0.15) and (abs(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y -
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y) < 0.03) and (abs(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x -
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x) < 0.1) and (abs(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y -
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y) < 0.03) and (abs(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y -
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y) < 0.03) and (abs(
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x -
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x) < 0.7):
                cv2.putText(img, "IP ATLIYOR", (180, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)  # Yeşil renk

    # FPS hesapla
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # FPS bilgisini görüntü üzerine ekle
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 65), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)  # Sarı renk

    # Görüntüyü göster
    cv2.imshow("Image", img)

    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Video akışını serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
