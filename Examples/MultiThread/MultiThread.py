import cv2
import face_recognition
import multiprocessing
import os

def cam1():
    video_capture = cv2.VideoCapture(0)

    #face lokasyon listesi
    face_locations = []

    while True:

        ret, frame = video_capture.read()


        rgb_frame = frame[:, :, ::-1]

        # tanımlanan yüzleri yukarda boş bıraktığımız liste lokasyonunun içine attı
        face_locations = face_recognition.face_locations(rgb_frame)

        for top, right, bottom, left in face_locations:
            # algılanan yüzün etrafına çerçeve çiz
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        # Çıkmak için q ya basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


def cam2():
    video_capture = cv2.VideoCapture("http://192.168.1.103:4747/video")


    face_locations = []

    while True:

        ret, frame = video_capture.read()


        rgb_frame = frame[:, :, ::-1]


        face_locations = face_recognition.face_locations(rgb_frame)

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

mp1=multiprocessing.Process(target=cam1)
mp2=multiprocessing.Process(target=cam2)

mp1.start()
mp2.start()
mp1.join()
mp2.join()