import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mnist_cnn_model.h5")

drawing = False
ix, iy = -1, -1
canvas = np.zeros((280, 280), dtype=np.uint8)

# Fare olayları
def draw(event, x, y, flags, param):
    global drawing, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (ix, iy), (x, y), 255, 15)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(canvas, (ix, iy), (x, y), 255, 15)

cv2.namedWindow("Draw a Digit")
cv2.setMouseCallback("Draw a Digit", draw)

while True:
    display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Draw a Digit", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("p"):  # Predict
        img = cv2.resize(canvas, (28, 28))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=(0, -1))  # (1, 28, 28, 1)

        prediction = model.predict(img)
        digit = np.argmax(prediction)

        print(f"Predicted Digit: {digit}")

        # Göster
        cv2.putText(display, f"Prediction: {digit}", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Draw a Digit", display)
        cv2.waitKey(1000)

    elif key == ord("c"):  # Clear canvas
        canvas = np.zeros((280, 280), dtype=np.uint8)

    elif key == ord("q"):  # Quit
        break

cv2.destroyAllWindows()
