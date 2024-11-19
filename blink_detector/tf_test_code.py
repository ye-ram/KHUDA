import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the trained model
model = load_model("2018_12_17_22_58_35.h5", compile=False)

# 2. Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download this file online

# 3. Define helper function to extract eye region
def get_eye_region(shape, eye_points):
    x = [shape.part(point).x for point in eye_points]
    y = [shape.part(point).y for point in eye_points]
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    return min_x, min_y, max_x - min_x, max_y - min_y

# 4. Define eye indices (landmarks for left and right eyes)
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

# 5. Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for i, face in enumerate(faces):
        # Get landmarks
        shape = predictor(gray, face)

        # Get left and right eye regions
        left_eye_rect = get_eye_region(shape, LEFT_EYE_POINTS)
        right_eye_rect = get_eye_region(shape, RIGHT_EYE_POINTS)

        # Extract eye regions from frame
        left_eye = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3],
                         left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
        right_eye = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3],
                          right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]

        # Preprocess eyes for the model
        def preprocess_eye(eye):
            try:
                eye = cv2.resize(eye, (34, 26))  # (width, height) 순서로 변경
                eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  # Grayscale 변환
                eye = eye / 255.0  # Normalize
                eye = eye.reshape(1, 26, 34, 1)  # (batch_size, height, width, channels)로 변경
                return eye
            except Exception as e:
                print(f"Preprocessing error: {e}")
                return None

        left_eye_input = preprocess_eye(left_eye)
        right_eye_input = preprocess_eye(right_eye)

        # Predict eye states
        left_eye_prediction = model.predict(left_eye_input) if left_eye_input is not None else [0]
        right_eye_prediction = model.predict(right_eye_input) if right_eye_input is not None else [0]

        # Interpret predictions
        left_eye_state = "Open" if left_eye_prediction > 0.5 else "Closed"
        right_eye_state = "Open" if right_eye_prediction > 0.5 else "Closed"

        # Display results
        cv2.putText(frame, f"Person {i + 1} Left Eye: {left_eye_state}", 
                    (face.left(), face.top() - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Person {i + 1} Right Eye: {right_eye_state}", 
                    (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Eye Blink Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
