# Import the necessary libraries
import cv2
import mediapipe as mp

def main():
    # Open the video stream
    cap = cv2.VideoCapture("video2.mp4")  # For video file; use 0 for webcam

    # Create the Mediapipe face detection class
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)

    # Mediapipe drawing utils object
    mp_draw = mp.solutions.drawing_utils

    while True:
        # Read a frame from the video stream
        success, img = cap.read()
        if not success:
            break

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with the face detection algorithm
        results = face_detection.process(img_rgb)

        # Draw detections on the image
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box

                # Get image dimensions
                h, w, _ = img.shape

                # Convert relative bounding box to pixel values
                bbox = (
                    int(bboxC.xmin * w),
                    int(bboxC.ymin * h),
                    int(bboxC.width * w),
                    int(bboxC.height * h)
                )

                # Draw the bounding box and confidence score
                cv2.rectangle(img, bbox, (0, 255, 255), 2)
                cv2.putText(
                    img,
                    f'{int(detection.score[0] * 100)}%',
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 255),
                    2
                )

        # Display the image
        cv2.imshow("Video", img)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
