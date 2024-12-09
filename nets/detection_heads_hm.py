import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from random import *
import torch
from torchsummary import summary

class DetectNet(nn.Module):
    def __init__(self, feature_dim=512, num_class=6,kernel_num=16):
        super(DetectNet, self).__init__()
        self.feature_dim = feature_dim
        self.num_class   = num_class

        self.fc_all = nn.Linear(self.feature_dim, 256)
        self.fc2 = nn.Linear(feature_dim, 256)
        self.fc3 = nn.Linear(256, 100)
        self.fc4 = nn.Linear(100, 1)

        self.deconv1 = nn.ConvTranspose2d(kernel_num, kernel_num, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(kernel_num, kernel_num, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(kernel_num, kernel_num, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(kernel_num, kernel_num, 3, stride=2, padding=1, output_padding=1)
        self.deconv0 = nn.ConvTranspose2d(1, kernel_num, 3, stride=2, padding=1, output_padding=1)

        self.conv2 = nn.Conv2d(kernel_num, kernel_num, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(kernel_num, kernel_num, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(kernel_num, kernel_num, (3, 3), padding=1)
        self.final_conv = nn.Conv2d(kernel_num, 1, (3, 3), padding=1)

        self.offset3 = nn.Linear(self.feature_dim, 100)
        self.offset4 = nn.Linear(100, 2)
        #  position heads
        self.position_est1 = nn.Linear(self.feature_dim, 256)
        self.position_est2 = nn.Linear(256, 128)
        self.position_est3 = nn.Linear(128, 1)

        #  classification heads
        self.cls1 = nn.Linear(self.feature_dim, 256)
        self.cls2 = nn.Linear(256, 128)
        self.cls3 = nn.Linear(128, self.num_class)

        # trajectory head
        self.traj1 = nn.Linear(self.feature_dim, 256)
        self.traj2 = nn.Linear(256, 128)
        self.traj3 = nn.Linear(128,30)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
    def forward(self, f_all):

        # Predict class
        class_detection = F.relu(self.cls1(f_all))
        # class_detection = self.dropout2(class_detection)
        class_detection =F.relu( self.cls2(class_detection))
        class_detection = self.cls3(class_detection)

        # trajectory detection
        trajectory = F.relu(self.traj1(f_all))
        # trajectory = self.dropout3(trajectory)
        trajectory  = F.relu(self.traj2(trajectory))
        trajectory  = self.traj3(trajectory)
        trajectory  = trajectory .view(-1, 10, 3)


        p = F.relu(self.fc2(f_all))
        p = F.relu(self.fc3(p))
        p = self.fc4(p)

        # Predict offset
        o = F.relu(self.offset3(f_all))
        o = self.offset4(o)

        # Predict heatmap
        feature_tft = F.relu(self.fc_all(f_all))
        feature_tft = feature_tft.view(-1, 1, 16, 16)
        x = F.relu(self.deconv0(feature_tft))

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.sigmoid(self.final_conv(x))
        x = x.view(-1, 512, 512)

        return x,o,p,class_detection,trajectory


