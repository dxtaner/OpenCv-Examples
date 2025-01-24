# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import numpy as np
import math

def find_angle(img, p1, p2, p3, lm_list, draw=True):
    """
    Üç nokta arasındaki açıyı bulur ve isteğe bağlı olarak resmi üzerine çizer.

    Parametreler:
    - img: Üzerinde çizim yapılacak resim
    - p1, p2, p3: Üç noktanın indeksleri
    - lm_list: Noktaların konumlarını içeren işaretler listesi
    - draw: Noktalar ve açıyı resme çizip çizmeme durumu

    Döndürür:
    - angle: Hesaplanan açı
    """
    x1, y1 = lm_list[p1][1:]
    x2, y2 = lm_list[p2][1:]
    x3, y3 = lm_list[p3][1:]
    
    # Açıyı hesapla
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle = angle + 360 if angle < 0 else angle

    if draw:
        # Noktalar ve açı üzerine çizim yap
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 3)
        cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (x3, y3), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (x1, y1), 15, (0, 255, 255))
        cv2.circle(img, (x2, y2), 15, (0, 255, 255))
        cv2.circle(img, (x3, y3), 15, (0, 255, 255))
        cv2.putText(img, str(int(angle)), (x2 - 40, y2 + 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    return angle

# Video akışını aç
cap = cv2.VideoCapture("video1.mp4")  # Video dosyasını belirtin veya 0 ile kamerayı kullanın

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

direction = 0
count = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    lm_list = []
    
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy])
    
    if lm_list:
        # Şınav sayacı
        angle = find_angle(img, 11, 13, 15, lm_list)
        percentage = np.interp(angle, (185, 245), (0, 100))
        
        if percentage == 100:
            if direction == 0:
                count += 0.5
                direction = 1
                
        if percentage == 0:
            if direction == 1:
                count += 0.5
                direction = 0
        
        cv2.putText(img, str(int(count)), (45, 125), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 7)
        
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
