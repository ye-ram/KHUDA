import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
import time

# 1. Load the trained model
model = load_model("model.h5", compile=False)

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
'''
# 해상도에 따라 변경
print("Default Resolution: {}x{}".format(
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
))
print("Set Resolution: {}x{}".format(
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
))
'''
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width 설정
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height 설정

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Set target width (maintain aspect ratio)
TARGET_WIDTH = 1280  # Desired width

# Variables to track eye state
capture_ready = False
show_alert = False
countdown_value = None
countdown_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get original frame dimensions
    height, width = frame.shape[:2]

    # Calculate new height while maintaining aspect ratio
    aspect_ratio = height / width
    new_height = int(TARGET_WIDTH * aspect_ratio)

    # Resize the frame to target dimensions
    resized_frame = cv2.resize(frame, (TARGET_WIDTH, new_height))

    # Convert frame to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    all_eyes_open = True  # Flag to check if all detected faces have both eyes open

    for i, face in enumerate(faces):
        # Get landmarks
        shape = predictor(gray, face)

        # Get left and right eye regions
        left_eye_rect = get_eye_region(shape, LEFT_EYE_POINTS)
        right_eye_rect = get_eye_region(shape, RIGHT_EYE_POINTS)

        # Extract eye regions from frame
        left_eye = resized_frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3],
                                 left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
        right_eye = resized_frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3],
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

        if left_eye_state != "Open" or right_eye_state != "Open":
            all_eyes_open = False

        # Draw a rectangle around the face
        cv2.rectangle(resized_frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

        # Display results for each person
        cv2.putText(resized_frame, f"Person {i + 1} Left Eye: {left_eye_state}", 
                    (face.left(), face.top() - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(resized_frame, f"Person {i + 1} Right Eye: {right_eye_state}", 
                    (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Check if all eyes are open and start countdown immediately
    if all_eyes_open and len(faces) > 0:
        if not capture_ready:
            countdown_start_time = time.time()
            capture_ready = True
    else:
        capture_ready = False
        show_alert = False

    if capture_ready:
        elapsed_time = time.time() - countdown_start_time
        if elapsed_time <= 1:
            countdown_value = 2
        elif elapsed_time <= 2:
            countdown_value = 1
        elif elapsed_time > 2:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, resized_frame)
            print(f"사진이 저장되었습니다: {filename}")
            break  # Exit the loop after saving the photo

        if countdown_value:
            cv2.putText(resized_frame, f"{countdown_value}", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

    # Show the resized frame
    cv2.imshow("Eye Blink Detection", resized_frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()

# "Photo has been taken!"
# "Eye Blink Detection"
