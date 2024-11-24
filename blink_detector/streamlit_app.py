# streamlit ver1
import streamlit as st
import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
import warnings

warnings.filterwarnings("ignore")

# 경로 설정
ALBUM_DIR = "album"
if not os.path.exists(ALBUM_DIR):
    os.makedirs(ALBUM_DIR)

# 모델 및 Dlib 로드
model = load_model("model.h5", compile=False)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 눈 영역 추출 함수
def get_eye_region(shape, eye_points):
    x = [shape.part(point).x for point in eye_points]
    y = [shape.part(point).y for point in eye_points]
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    return min_x, min_y, max_x - min_x, max_y - min_y

# 눈 예측 함수
def preprocess_eye(eye):
    try:
        eye = cv2.resize(eye, (34, 26))
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        eye = eye / 255.0
        eye = eye.reshape(1, 26, 34, 1)
        return eye
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

# 사진 저장 함수
def save_photo(frame):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(ALBUM_DIR, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Photo saved at: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving photo: {e}")
        return None

# 사진 삭제 함수
def delete_photo(filename):
    if os.path.exists(filename):
        os.remove(filename)
        return True
    return False

# Streamlit UI
st.set_page_config(page_title="Eye Blink Detection", page_icon="👁️", layout="wide")
st.title("👁️ Eye Blink Detection")
st.sidebar.header("📋 메뉴")

menu = st.sidebar.radio("기능 선택", ("📸 자동 촬영", "📸 수동 촬영", "🖼️ 앨범 보기"))

if menu == "📸 자동 촬영":
    st.header("📸 자동 촬영")
    start_capture = st.button("🚀 자동 촬영 시작", key="auto_start")

    if start_capture:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        capture_ready = False
        countdown_start_time = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("카메라를 읽을 수 없습니다.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            all_eyes_open = True
            for face in faces:
                shape = predictor(gray, face)
                left_eye_rect = get_eye_region(shape, list(range(36, 42)))
                right_eye_rect = get_eye_region(shape, list(range(42, 48)))

                left_eye = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3],
                                 left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
                right_eye = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3],
                                  right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]

                left_eye_input = preprocess_eye(left_eye)
                right_eye_input = preprocess_eye(right_eye)

                left_eye_prediction = model.predict(left_eye_input) if left_eye_input is not None else [0]
                right_eye_prediction = model.predict(right_eye_input) if right_eye_input is not None else [0]

                if left_eye_prediction <= 0.5 or right_eye_prediction <= 0.5:
                    all_eyes_open = False

            # 화면에 눈 상태 표시
            if all_eyes_open:
                cv2.putText(frame, "All eyes are open!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Someone's eyes are closed!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if len(faces) == 0:
                cv2.putText(frame, "No face detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if all_eyes_open and len(faces) > 0:
                if not capture_ready:
                    countdown_start_time = time.time()
                    capture_ready = True

                elapsed_time = time.time() - countdown_start_time
                if elapsed_time > 2:
                    filename = save_photo(frame)
                    st.image(filename, caption="📸 촬영된 사진", use_column_width="auto")
                    st.success(f"✅ 사진이 저장되었습니다: {filename}")
                    break
            else:
                capture_ready = False

            st_frame.image(frame, channels="BGR")

        cap.release()

elif menu == "📸 수동 촬영":
    st.header("📸 수동 촬영")

    cap = cv2.VideoCapture(0)
    st_frame = st.empty()

    if st.button("📸 Capture Photo"):
        ret, frame = cap.read()
        if ret:
            filename = save_photo(frame)
            if filename:
                st.image(filename, caption="📷 Captured Photo", use_column_width="auto")
                st.success(f"✅ 사진이 저장되었습니다: {filename}")
            else:
                st.error("❌ 사진 저장에 실패했습니다.")
        else:
            st.error("❌ 캠에서 영상을 읽을 수 없습니다.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("카메라를 읽을 수 없습니다.")
            break

        # 화면에 눈 상태 표시
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        all_eyes_open = True

        if len(faces) > 0:
            for face in faces:
                shape = predictor(gray, face)
                left_eye_rect = get_eye_region(shape, list(range(36, 42)))
                right_eye_rect = get_eye_region(shape, list(range(42, 48)))

                left_eye = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3],
                                 left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
                right_eye = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3],
                                  right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]

                left_eye_input = preprocess_eye(left_eye)
                right_eye_input = preprocess_eye(right_eye)

                if left_eye_input is not None and right_eye_input is not None:
                    left_eye_prediction = model.predict(left_eye_input)
                    right_eye_prediction = model.predict(right_eye_input)

                    if left_eye_prediction <= 0.5 or right_eye_prediction <= 0.5:
                        all_eyes_open = False

            if all_eyes_open:
                cv2.putText(frame, "All eyes are open!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Someone's eyes are closed!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        st_frame.image(frame, channels="BGR")

    cap.release()

elif menu == "🖼️ 앨범 보기":
    st.header("🖼️ 앨범 보기")
    image_files = [os.path.join(ALBUM_DIR, f) for f in os.listdir(ALBUM_DIR) if f.endswith(".jpg")]
    if image_files:
        cols = st.columns(3)
        for idx, img_file in enumerate(image_files):
            with cols[idx % 3]:
                st.image(img_file, caption=f"🖼️ {os.path.basename(img_file)}", use_column_width=True)
                if st.button("🗑️ 삭제", key=f"delete_{img_file}"):
                    if delete_photo(img_file):
                        st.success(f"✅ {os.path.basename(img_file)} 삭제됨")
                        st.experimental_rerun()
    else:
        st.warning("📂 저장된 사진이 없습니다.")
