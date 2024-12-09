import torch.nn as nn
import torch.nn.functional as F
import torch
from nets.audio_net import AudioNet
from nets.comparison.detection_heads import DetectNet


class AudioNet_detect(nn.Module):
    def __init__(self, dropout_rate=0.3, kerel_num=16, feature_dim=512,classes=6):
        super(AudioNet_detect, self).__init__()
        self.audionet             = AudioNet(dropout_rate, kerel_num, feature_dim)
        self.detectnet            = DetectNet(feature_dim,classes)

    def forward(self, x):
        feature_tf = self.audionet(x)
        position,class_detection,trajectory = self.detectnet(feature_tf)
        return position,class_detection

