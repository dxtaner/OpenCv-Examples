from cvzone.PlotModule import LivePlot
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
import math
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı. Lütfen bağlantıyı kontrol edin.")
    exit()

detector = FaceDetector(minDetectionCon=0.85, modelSelection=0)

xPlot = LivePlot(w=1200, yLimit=[0, 500], interval=0.01, char='X')
sinPlot = LivePlot(w=1200, yLimit=[-100, 100], interval=0.01, char="S")
xSin = 0

while True:
    success, img = cap.read()
    if not success:
        print("Kameradan görüntü alınamadı. Lütfen kamerayı kontrol edin.")
        break

    img, bboxs = detector.findFaces(img, draw=False)
    val = 0

    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)
            val = center[0]

            cvzone.cornerRect(img, (x, y, w, h), colorR=(0, 255, 0), colorC=(0, 255, 0), t=2)
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, f'{score}%', (x, y - 10), colorR=(0, 255, 0), colorB=(0, 0, 0))

    imgPlot = xPlot.update(val)
    xSin += 1
    if xSin == 360: xSin = 0
    imgPlotSin = sinPlot.update(int(math.sin(math.radians(xSin)) * 100))

    imgPlot = cv2.applyColorMap(imgPlot, cv2.COLORMAP_JET)
    imgPlotSin = cv2.applyColorMap(imgPlotSin, cv2.COLORMAP_JET)

    imgStack = cvzone.stackImages([img, imgPlot], 2, 1)

    cv2.imshow("Image Plot", imgPlot)
    cv2.imshow("Image Sin Plot", imgPlotSin)
    cv2.imshow("Image Stack", imgStack)
    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()