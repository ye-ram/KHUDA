import torch
import torch.nn as nn
import torch.nn.functional as F

'''

Class ArcFace
    1. def __init__(self, in_dim, out_dim, s, m):
        - s and m are parameters derived from "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".
        - Matrix W:
            1) The matrix W has dimensions in_dim x out_dim.
            2) W is initialized using Xavier initialization.
            3) in_dim: Dimensionality of the tensor resulting from flattening the forward pass of VGG19.
            4) out_dim: Number of classes.
            
    2. def forward(self, x):
        - the forward pass of the ArcFace model.

'''
# ArcFace
# 클래스간 각도를 통해 차이를 주어 서로 다른 클래스간(얼굴 특징 벡터 간)에는 더 큰 격차를 만드는 방법
# 얼굴 임베딩(embedding) 공간에서 서로 다른 사람들의 얼굴 벡터가 더 멀어지고, 같은 사람의 얼굴 벡터는 더 가까워지게된다. 
# 정규화된 소프트맥스 손실 함수를 기반으로 한다. 
# 얼굴 데이터가 많을 때 뛰어난 성능을 보인다. 

class ArcFace(nn.Module):
    def __init__(self, in_dim, out_dim, s, m): 
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))

        # s: 스케일링 파라미터. ArcFace에서는 일반적으로 학습을 안정화하고 성능을 높이기 위해 사용
        # m: 각도 마진(angular margin)으로, 얼굴 임베딩 간의 차이를 더 명확히 하기 위해 사용
        # W: 학습 가능한 가중치 매트릭스 W 정의. 이 매트릭스는 얼굴 임베딩 벡터와 매칭될 분류기의 가중치를 나타냄. 

        nn.init.kaiming_uniform_(self.W)
        # 가중치 매트릭스 W를 카이밍(Kaiming) 초기화로 설정.  딥러닝에서 자주 쓰이는 초기화 방식으로, ReLU와 궁합이 좋다.

    def forward(self, x): 
        # x: 얼굴 임베딩 벡터. 신경망의 출력으로 전달되는 벡터.
        normalized_x = F.normalize(x, p=2, dim=1)
        normalized_W = F.normalize(self.W, p=2, dim=0)
        # F.normalize(): 입력 벡터 x와 가중치 매트릭스 W를 L2 정규화한다. 
        # ArcFace는 임베딩 벡터와 가중치 벡터를 정규화한 후, 코사인 유사도를 계산하는데, 이 과정을 통해 크기 차이를 무시하고 벡터의 방향만 비교할 수 있게 된다.
        # p=2: L2 정규화. 벡터의 길이가 1이 되도록 변환
        # dim=1, dim=0: x는 배치의 차원을 따라, W는 출력 차원을 따라 정규화
    
        cosine = torch.matmul(normalized_x.view(normalized_x.size(0), -1), normalized_W)
        # torch.matul()은 두 텐서 간의 행렬 곱셈하는 함수.
        # normalized_x는 크기가 (btch_size, in_dim)인 2차원 텐서
        # normalized_z는 크기가 (in_dim, out_dim)인 2차원 텐서

        # Using torch.clamp() to ensure cosine values are within a safe range,
        # preventing potential NaN losses.
        
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        # torch.acos(): 아크 코사인 함수. 코사인 값을 각도로 변환해 각도를 구함. 
        # torch.clamp(): 입력값을 코사인 함수의 정의역 [-1, 1]을 벗어나지 않도록 함. 

        probability = self.s * torch.cos(theta+self.m)
        
        return probability