import torch.nn as nn
from nets.comparison.detection_heads import DetectNet
from nets.visual_net import VisualNet

"""
This script utilizes Resnet as the backbone as the visual network for feature extraction
"""


class ResNet_VisualNet(nn.Module):
    def __init__(self, feature_dim=256):
        super(ResNet_VisualNet, self).__init__()
        self.feature_dim = feature_dim
        self.visualnet            = VisualNet(self.feature_dim)
        self.detectnet            = DetectNet(self.feature_dim,self.classes)

    def forward(self,y):
        visual_feature        = self.visualnet(y)
        position,class_detection,trajectory = self.detectnet(visual_feature)
        return position,class_detection
