# VGG19와 ArcFace를 결합해 입력 이미지를 처리

from Backbone import VGG19
from ArcFace import ArcFace
import torch.nn as nn

class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.VGG19 = VGG19()
        self.ArcFace = ArcFace(in_dim = 25088, out_dim = 20, s = 64, m = 0.6)
    
    def forward(self, x):
        x = self.VGG19(x)
        x = self.ArcFace(x)
        
        return x