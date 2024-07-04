import cv2
import cvlib as cv
import time

kamera = cv2.VideoCapture(0)
kamera.set(3, 1920)
kamera.set(4, 1080)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


while kamera.isOpened():
    ret, frame = kamera.read()

    faces, confidence = cv.detect_face(frame)

    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_color)

        if not eyes:
            t = 1
            while not eyes:
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

    cv2.imshow("Webcam", frame)

    if (cv2.waitKey(30) & 0xFF == ord('q')):
        break

kamera.release()
cv2.destroyAllWindows()
