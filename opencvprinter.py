import cv2
import numpy as np

class Paint:
    def __init__(self):
        self.image = np.zeros((500, 500, 3), np.uint8)
        self.color = (0, 0, 0)

    def run(self):
        cv2.namedWindow('Paint')
        cv2.setMouseCallback('Paint', self.onMouse)

        while True:
            cv2.imshow('Paint', self.image)
            key = cv2.waitKey(10)
            if key == 27:
                break

        cv2.destroyAllWindows()

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            self.paintPixel(x, y)

    def paintPixel(self, x, y):
        self.image[y, x] = self.color


Paint().run()