import cv2
import mediapipe as mp
import time


def main():
    # Video kaynağını aç
    cap = cv2.VideoCapture(0)  # Kamera kullanımı için 0; video dosyası için "video1.mp4"
    pTime = 0

    # Mediapipe yüz tanıma modülleri
    mp_face_detection = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)

    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                draw_detection(img, detection, mp_draw)

        # FPS hesapla ve ekrana yazdır
        pTime = display_fps(img, pTime)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_detection(img, detection, mp_draw):
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = img.shape
    bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
            int(bboxC.width * iw), int(bboxC.height * ih))

    cv2.rectangle(img, bbox, (255, 0, 255), 2)
    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


def display_fps(img, pTime):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    return pTime


if __name__ == "__main__":
    main()
