import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class VisualNet(nn.Module):
    def __init__(self, feature_dim=512):
        super(VisualNet, self).__init__()
        self.feature_dim = feature_dim
        # Visual Net
        self.resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_feature = nn.Sequential(*list(self.resnet_model.children())[:-2])
        # add additional conv following the resnet feature
        self.convres = nn.Conv2d(2048, 256, (3, 3), padding=1)
        self.convvis = nn.Conv2d(256,32, (3, 3), padding=1)
        self.fcvis = nn.Linear(2048, self.feature_dim)

    def forward(self, y):
        [b, c, row, col] = y.size()
        y = self.resnet_feature(y)
        y = F.relu(self.convres(y))
        y = F.relu(self.convvis(y))
        fi = y.contiguous().view(b,-1)
        fi = F.relu(self.fcvis(fi))
        return fi
