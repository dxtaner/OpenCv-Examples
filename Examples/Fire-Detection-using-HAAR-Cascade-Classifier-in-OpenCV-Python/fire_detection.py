# Import necessary libraries
import numpy as np
import cv2
# import serial
import time

# Load the cascade classifier for fire detection
fire_cascade = cv2.CascadeClassifier('cascade.xml')

# Open a video capture device
cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    ret, img = cap.read()  # Capture a frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    # Detect fire in the image
    fire = fire_cascade.detectMultiScale(img, 12, 5)

    for (x, y, w, h) in fire:
        # Highlight the area of the image with fire
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        print('Fire is detected..!' + str(count))
        count += 1
        # ser1.write(str.encode('p'))  # Write 'p' on the serial COM port to Arduino
        time.sleep(0.2)  # Wait

    cv2.imshow('img', img)
    # ser1.write(str.encode('s'))  # Write 's' if there is no fire
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

# ser1.close()
cap.release()
cv2.destroyAllWindows()
