# Import necessary libraries
import cv2
import time
import mediapipe as mp

def main():
    # Open the camera stream
    cap = cv2.VideoCapture(0)  # 0 for the default webcam

    # Create the Mediapipe face mesh class
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

    # Mediapipe drawing utils object
    mp_draw = mp.solutions.drawing_utils
    draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    # Timing for FPS calculation
    prev_time = 0

    while True:
        # Read a frame from the camera stream
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with the face mesh algorithm
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh landmarks on the image
                mp_draw.draw_landmarks(
                    img,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    draw_spec,
                    draw_spec
                )

                # Print landmark coordinates
                for id, lm in enumerate(face_landmarks.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(f"ID: {id}, X: {cx}, Y: {cy}")

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Display the image
        cv2.imshow("Face Mesh", img)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Release the camera stream and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
