import torch.nn as nn
import torch.nn.functional as F
import torch
from nets.audio_net import AudioNet
from nets.visual_net import VisualNet
from nets.comparison.detection_heads import DetectNet

"""
This network is the replementation of the paper: AV-PedAware: Self-Supervised Audio-Visual Fusion for Dynamic Pedestrian Awareness
"""

class AVPed(nn.Module):
    def __init__(self, feature_dim=256,classes=6):
        super(AVPed, self).__init__()
        self.feature_dim = feature_dim
        self.classes     = classes
        self.detectnet            = DetectNet(self.feature_dim,self.classes)
        self.audionet             = AudioNet(feature_dim=self.feature_dim)
        self.visualnet            = VisualNet(self.feature_dim)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout_att1 = nn.Dropout1d(0.4)


    def forward(self, x, y):
        [ba, ca, rowa, cola] = x.size()
        audio_feature        = self.audionet(x)
        audio_feature = audio_feature.view(ba,1,-1)
        visual_feature        = self.visualnet(y)
        visual_feature   = visual_feature.reshape(ba,1,-1)
        f_all = torch.cat([audio_feature, visual_feature], dim=1)

        fi = visual_feature.view(ba,-1)
        detect = F.relu(self.fc1(fi))
        detect = self.fc2(detect)
        detect_soft = F.softmax(detect,-1)
        detect_expanded = detect_soft.unsqueeze(-1)
        f_all = f_all * detect_expanded
        f_all = torch.sum(f_all, 1)
        position,class_detection,trajectory = self.detectnet(f_all)

        return position,detect,class_detection,trajectory
