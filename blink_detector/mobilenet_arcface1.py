import os
from shutil import copyfile
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import zipfile
import torch.optim as optim

# CUDA 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 중인 장치: {device}")
if device == 'cuda':
    print(f"사용 가능한 GPU: {torch.cuda.get_device_name(0)}")

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# Step 1: CelebA 데이터셋 분류
zip_path = '/content/drive/MyDrive/celeba-dataset.zip'
# /content/drive/MyDrive/celeba-dataset.zip
extract_path = '/content/img_align_celeba/'

# 압축 해제
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("압축 해제 완료.")
else:
    print("이미 압축 해제됨.")

label_file = '/content/img_align_celeba/list_attr_celeba.csv'
img_folder = '/content/img_align_celeba/img_align_celeba/img_align_celeba'
partition_file = '/content/img_align_celeba/list_eval_partition.csv'

# 저장 폴더 생성
os.makedirs('/content/data/train/closed', exist_ok=True)
os.makedirs('/content/data/train/open', exist_ok=True)
os.makedirs('/content/data/test/closed', exist_ok=True)
os.makedirs('/content/data/test/open', exist_ok=True)

# 속성 및 분할 파일 읽기
attr_df = pd.read_csv(label_file)
partition_df = pd.read_csv(partition_file)

# 데이터 병합
data = pd.merge(attr_df, partition_df, on='image_id')

# Train/Test 데이터 분리 및 복사
for _, row in data.iterrows():
    img_name = row['image_id']
    partition = row['partition']
    is_closed = row['Eyeglasses'] == -1  # Eyeglasses 속성 기준 분류

    if partition == 0:  # Train 데이터
        dest_folder = '/content/data/train/closed' if is_closed else '/content/data/train/open'
    elif partition == 2:  # Test 데이터
        dest_folder = '/content/data/test/closed' if is_closed else '/content/data/test/open'
    else:
        continue  # Validation 데이터는 무시

    src_path = os.path.join(img_folder, img_name)
    dest_path = os.path.join(dest_folder, img_name)

    try:
        copyfile(src_path, dest_path)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {src_path}")

# Step 2: 데이터셋 클래스 정의
class EyeStateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 폴더 구조: root_dir/closed, root_dir/open
        for label, sub_dir in enumerate(['closed', 'open']):
            folder = os.path.join(root_dir, sub_dir)
            for img_name in os.listdir(folder):
                self.image_paths.append(os.path.join(folder, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_with_augmentation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 및 데이터로더 준비
train_dataset = EyeStateDataset(root_dir='/content/data/train', transform=transform_with_augmentation)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = EyeStateDataset(root_dir='/content/data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

# ArcFace 정의 (기존 코드 유지)
class ArcFace(nn.Module):
    def __init__(self, in_dim, out_dim, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, labels):
        x = nn.functional.normalize(x, dim=1)
        weight = nn.functional.normalize(self.weight, dim=1)

        cosine = torch.matmul(x, weight.t())
        theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))
        arc_margin = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine).scatter(1, labels.view(-1, 1), 1.0)
        logits = one_hot * arc_margin + (1.0 - one_hot) * cosine
        logits *= self.s
        return logits

# MobileNet + ArcFace 모델 정의 (기존 코드 유지)
class EyeStateModelWithArcFace(nn.Module):
    def __init__(self, arcface_in_dim, num_classes=2, use_arcface=True):
        super(EyeStateModelWithArcFace, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(self.mobilenet.features.children()),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(self.mobilenet.last_channel, arcface_in_dim)
        self.arcface = ArcFace(in_dim=arcface_in_dim, out_dim=num_classes)
        self.use_arcface = use_arcface

    def forward(self, x, labels=None):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        features = self.fc(x)

        if self.use_arcface and labels is not None:
            logits = self.arcface(features, labels)
        else:
            logits = nn.functional.linear(features, self.arcface.weight)
        return logits

# 학습 설정
arcface_in_dim = 1280
model = EyeStateModelWithArcFace(arcface_in_dim=arcface_in_dim, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Epoch 및 TensorBoard 초기화
EPOCH = 100
writer = SummaryWriter()

# 정확도와 손실 계산 함수
def compute_accuracy_and_loss(device, model, data_loader):
    model.eval()
    loss, example_num, correct_num = 0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images, labels)
            loss += criterion(logits, labels).item()

            _, preds = torch.max(logits, 1)
            example_num += labels.size(0)
            correct_num += (preds == labels).sum().item()

    return loss / example_num, (correct_num / example_num) * 100

# 가중치 저장 함수
def save_weight(model, path):
    torch.save(model.state_dict(), path)

# 학습 루프
start_time = time.time()

for epoch in range(EPOCH):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:  # 매 10번째 배치마다 로그 출력
            print(f"Epoch: {epoch + 1}/{EPOCH}, Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # 훈련 손실 및 정확도 계산
    train_loss, train_acc = compute_accuracy_and_loss(device, model, train_loader)

    # 테스트 손실 및 정확도 계산
    test_loss, test_acc = compute_accuracy_and_loss(device, model, test_loader)

    # TensorBoard에 기록
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", test_acc, epoch)

    # 로그 출력
    print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Epoch {epoch + 1}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # 모델 가중치 저장 (매 10 에포크마다)
    if (epoch + 1) % 10 == 0:
        save_weight(model, f"/content/drive/MyDrive/EyeStateModel_Epoch{epoch + 1}.pth")

elapsed_time = (time.time() - start_time) / 60
print(f"Total Training Time: {elapsed_time:.2f} minutes")

writer.close()

# 모델 평가
model.eval()
model.use_arcface = False  # ArcFace 비활성화
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 평가 결과 계산
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

