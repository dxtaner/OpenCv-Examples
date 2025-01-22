import cv2
import numpy as np


def region_of_interest(image, vertices):
    """
    Creates a mask for a region of interest in the image and applies it.

    Args:
    image (np.ndarray): The input image.
    vertices (np.ndarray): The vertices of the polygon defining the region of interest.

    Returns:
    np.ndarray: The masked image.
    """
    mask = np.zeros_like(image)  # Create a black mask with the same dimensions as the image
    match_mask_color = 255  # Define the color intensity for the mask

    cv2.fillPoly(mask, vertices, match_mask_color)  # Fill the mask with white polygons
    masked_image = cv2.bitwise_and(image, mask)  # Apply the mask to the image
    return masked_image


def draw_lines(image, lines):
    """
    Draws lines on an image.

    Args:
    image (np.ndarray): The input image.
    lines (list): List of lines to be drawn.

    Returns:
    np.ndarray: The image with lines drawn.
    """
    image = np.copy(image)  # Create a copy of the image
    blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)  # Create a blank black image

    if lines is not None:  # Ensure lines is not None
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)  # Draw green lines

    image = cv2.addWeighted(image, 0.7, blank_image, 1, 0.0)  # Combine the original image and the lines image
    return image


def process(image):
    """
    Processes the image to detect lanes and draw them.

    Args:
    image (np.ndarray): The input image.

    Returns:
    np.ndarray: The image with lanes drawn.
    """
    height, width = image.shape[:2]  # Get the height and width of the image
    region_of_interest_vertices = [(width * 0.7, 0), (width, height),
                                   (0, height)]  # Define vertices for region of interest

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    canny_img = cv2.Canny(img_gray, 370, 150)  # Apply Canny edge detection
    cropped_image = region_of_interest(canny_img, np.array([region_of_interest_vertices],
                                                           np.int32))  # Apply region of interest mask
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=220, lines=np.array([]),
                            minLineLength=150, maxLineGap=5)  # Apply Hough line transformation

    image_with_lines = draw_lines(image, lines)  # Draw detected lines on the image
    return image_with_lines


# Open the video stream
cap = cv2.VideoCapture("video2.mp4")  # Open the video file for processing

while True:
    success, img = cap.read()  # Read a frame from the video
    if not success:
        break

    img = process(img)  # Process the frame to detect and draw lanes
    cv2.imshow("Lane Detection", img)  # Display the processed frame

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit the loop if 'q' key is pressed
        break

cap.release()  # Release the video stream
cv2.destroyAllWindows()  # Close all OpenCV windows
