import pafy
import cv2

url = 'https://www.youtube.com/watch?v=5MsFBp6joP4'

vPaffy = pafy.new(url)
#play = vPaffy.getbest(preftype="webm")
play = vPaffy.getbest()

cam = cv2.VideoCapture(play.url)
while True:
    _, video = cam.read()
    griton = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)

    cv2.imshow("video",video)
    cv2.imshow("griton",griton)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
