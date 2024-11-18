import cv2
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms
from PIL import Image
import time

# YOLO 얼굴 감지 모델 초기화 (YOLOv8-Face)
yolo_model = YOLO('yolov8n.pt')  # YOLOv8 얼굴 감지 전용 모델

# MobileNet 학습된 눈 감지 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EyeStateModel(torch.nn.Module):
    def __init__(self):
        super(EyeStateModel, self).__init__()
        self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
        self.mobilenet.classifier[1] = torch.nn.Linear(self.mobilenet.last_channel, 2)

    def forward(self, x):
        return self.mobilenet(x)

eye_model = EyeStateModel()
eye_model.load_state_dict(torch.load('eye_state_model.pth', map_location=device))
eye_model.to(device)
eye_model.eval()

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # MobileNet 정규화
])

# OpenCV로 웹캠 연결
cap = cv2.VideoCapture(0)

# OpenCV 해상도 조정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 상태 추적 변수
all_open_count = 0  # 모든 사람이 눈을 뜬 상태를 유지한 프레임 수
required_open_frames = 30  # 30 프레임 동안 눈을 뜬 상태를 유지해야 메시지 표시
timer_active = False  # 타이머 상태 추적
timer_start_time = 0
capture_saved = False  # 이미지 캡처 상태 추적

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 영상을 가져올 수 없습니다.")
            break

        results = yolo_model(frame)  # YOLO 실행 결과
        all_open = True  # 모든 사람이 눈을 뜬 상태인지 확인하는 플래그

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0])

                # 신뢰도 및 크기 필터링
                if cls == 0 and conf > 0.5 and (x2 - x1) > 50 and (y2 - y1) > 50:
                    # 얼굴 영역 추출
                    face = frame[int(y1):int(y2), int(x1):int(x2)]
                    if face.size != 0:
                        pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        input_tensor = transform(pil_face).unsqueeze(0).to(device)
                        outputs = eye_model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probs, 1)
                        
                        eye_state = "Closed" if predicted.item() == 0 else "Open"

                        # 신뢰도가 낮으면 "Unknown"으로 처리
                        if confidence.item() < 0.7:
                            eye_state = "Unknown"
                        elif eye_state == "Closed":
                            all_open = False

                        # 바운딩 박스와 눈 상태 출력
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{eye_state} ({confidence.item():.2f})", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # 모든 사람이 눈을 뜬 상태인 경우
        if all_open:
            all_open_count += 1
        else:
            all_open_count = 0

        # 일정 프레임 동안 모두 눈을 뜬 상태일 경우 메시지 표시
        if all_open_count >= required_open_frames and not timer_active:
            cv2.putText(frame, "All eyes are open. Preparing to capture...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            timer_active = True
            timer_start_time = time.time()  # 타이머 시작 시간 기록

        # 타이머 실행
        if timer_active:
            elapsed_time = time.time() - timer_start_time
            countdown = 3 - int(elapsed_time)  # 3-2-1 카운트다운
            if countdown > 0:
                cv2.putText(frame, f"Capturing in {countdown}...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                if not capture_saved:  # 이미지가 아직 저장되지 않았다면
                    capture_saved = True
                    cv2.imwrite('captured_frame.jpg', frame)  # 프레임 저장
                    print("이미지가 저장되었습니다: captured_frame.jpg")

                # 타이머 종료 후 다시 초기화
                timer_active = False
                all_open_count = 0
                capture_saved = False  # 캡처 상태 초기화

        # 프레임 출력
        cv2.imshow('YOLO Eye Detection', frame)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
