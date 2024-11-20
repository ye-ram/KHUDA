import cv2
import dlib
import numpy as np
import time
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# 1. Load the trained model
model = load_model("resnet_model.h5", compile=False)

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

# Tracking variables
eye_open_for_capture = []  # List to track the open duration for each face
all_open_count = 0  # To track how long everyone has eyes open
required_open_frames = 30  # Number of frames for 1 second at 30 FPS
timer_active = False
capture_saved = False

# Font settings for Pillow (Make sure to have a valid path to a Korean font)
font_path = "path_to_your_font/NanumGothic.ttf"  # Use a Korean font file
font = ImageFont.truetype(font_path, 24)  # Set font size to 32

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    all_open = True  # Assume all eyes are open initially
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

        # Check if any eye is closed
        if left_eye_state == "Closed" or right_eye_state == "Closed":
            all_open = False

    # All eyes open check
    if all_open:
        all_open_count += 1
    else:
        all_open_count = 0

    # When everyone has eyes open for required frames, show countdown
    if all_open_count >= required_open_frames and not timer_active:
        timer_active = True
        timer_start_time = time.time()  # Start the timer

    # Timer countdown logic
    if timer_active:
        elapsed_time = time.time() - timer_start_time
        countdown = 2 - int(elapsed_time)  # 2-1 countdown
        if countdown > 0:
            # Use Pillow to write Korean text on the frame
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            draw.text((50, 50), "모두 눈을 떴습니다. 캡처 준비 중...", font=font, fill=(0, 0, 255))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Countdown
            cv2.putText(frame, f"Capturing in {countdown}...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            if not capture_saved:  # Save the image if it hasn't been saved yet
                capture_saved = True
                cv2.imwrite('captured_frame.jpg', frame)  # Save the captured frame
                print("Image captured!")

            # Reset after the countdown
            timer_active = False
            all_open_count = 0
            capture_saved = False  # Reset capture state

    # Show the frame
    cv2.imshow('Eye Blink Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()


