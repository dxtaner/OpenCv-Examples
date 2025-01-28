import cv2
import numpy as np
from collections import deque

# Buffer size for tracking points
BUFFER_SIZE = 16
pts = deque(maxlen=BUFFER_SIZE)

# HSV range for blue color
define_blue_lower = (84, 98, 0)
define_blue_upper = (179, 255, 255)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture a frame from the webcam
    success, frame = cap.read()

    if not success:
        print("Failed to capture frame. Exiting...")
        break

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask for detecting blue objects
    mask = cv2.inRange(hsv, define_blue_lower, define_blue_upper)

    # Apply erosion and dilation to remove noise
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if contours:
        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Fit a minimum area rectangle to the largest contour
        rect = cv2.minAreaRect(largest_contour)
        ((x, y), (width, height), rotation) = rect

        # Format and display rectangle information
        rect_info = f"x: {x:.1f}, y: {y:.1f}, width: {width:.1f}, height: {height:.1f}, rotation: {rotation:.1f}"
        print(rect_info)

        # Get the rectangle box points and convert to integer
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Compute the centroid using moments
        moments = cv2.moments(largest_contour)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

        # Draw the rectangle and centroid on the frame
        cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (255, 0, 255), -1)

        # Display rectangle information on the frame
        cv2.putText(frame, rect_info, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Track the points and draw the trajectory
    pts.appendleft(center)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), 3)

    # Display the processed frames
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Mask", mask)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
