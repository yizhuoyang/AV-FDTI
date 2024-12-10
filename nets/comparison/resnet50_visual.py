import torch.nn as nn
from nets.comparison.detection_heads import DetectNet
from nets.visual_net import VisualNet
from torchinfo import summary
import torch
"""
This script utilizes Resnet as the backbone as the visual network for feature extraction
"""
class ResNet_VisualNet(nn.Module):
    def __init__(self, feature_dim=256, classes=6):
        super(ResNet_VisualNet, self).__init__()
        self.feature_dim = feature_dim
        self.visualnet            = VisualNet(self.feature_dim)
        self.detectnet            = DetectNet(self.feature_dim, classes)

    def forward(self,y):
        visual_feature        = self.visualnet(y)
        position,class_detection,_ = self.detectnet(visual_feature)
        return position,class_detection


if __name__ == "__main__":

    feature_dim = 512
    classes = 6
    model = ResNet_VisualNet(feature_dim=feature_dim, classes=classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_shape = (3,256,256)
    summary(model, input_size=(1, 3,256,256))

    batch_size = 1
    dummy_input = torch.randn(batch_size, *input_shape).to(device)
    position, class_detection = model(dummy_input)
    print("\nForward Pass Results:")
    print("Position Output Shape:", position.shape)
    print("Class Detection Output Shape:", class_detection.shape)
