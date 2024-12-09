import torch.nn as nn

"""
This network is the replementation of the paper: DroneChase: A Mobile and Automated Cross-Modality System for Continuous Drone Tracking
"""

class voranet(nn.Module):
    def __init__(self):
        super(voranet, self).__init__()
        # Block1
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        # Dense
        self.fc1 = nn.Linear(128, 32)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(32, 6)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), x.size(2), -1)
        x = x[:, :, :128]

        x, _ = self.lstm(x)

        x = self.relu4(self.fc1(x[:, -1, :]))
        c = self.fc3(x)
        x = self.fc2(x)

        return x,c


