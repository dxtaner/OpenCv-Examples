import cv2
import numpy as np

# Start the webcam
cap = cv2.VideoCapture(0)

# Define the HSV color range for the object (red in this example)
lower_red = np.array([160, 100, 100])
upper_red = np.array([179, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the red color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Reduce noise in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:  # Filter out small objects
            # Compute the center of the contour
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Draw a circle around the object
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)

            # Display coordinates
            cv2.putText(frame, f"X:{int(x)} Y:{int(y)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the result
    cv2.imshow("Object Tracking", frame)
    cv2.imshow("Mask", mask)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
