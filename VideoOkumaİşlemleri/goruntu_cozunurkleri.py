import cv2

kamera = cv2.VideoCapture(0)

def cozunurlukler_1080p():
    kamera.set(3, 1920)
    kamera.set(4, 1080)

def cozunurlukler_720p():
    kamera.set(3, 1280)
    kamera.set(4, 720)

def cozunurlukler_480p():
    kamera.set(3, 640)
    kamera.set(4, 480)

def cozunurlukleribelirle(width,height):
    kamera.set(3, width)
    kamera.set(4, height)

def skalamalama(frame,percent=75):
    witdh = int(frame.shape[1] * percent / 100)
    heigth = int(frame.shape[0] * percent / 100)

    boyut=(witdh,heigth)

    return cv2.resize(frame,boyut,interpolation=cv2.INTER_AREA)


cozunurlukleribelirle(400,500)

while True:
    ret, frame = kamera.read()
    frame75=skalamalama(frame,75)

    cv2.imshow("goruntu1",frame)
    cv2.imshow("goruntu2",frame75)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()