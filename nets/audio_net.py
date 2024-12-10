import torch.nn as nn
import torch.nn.functional as F
import torch


class AudioNet(nn.Module):
    def __init__(self, dropout_rate=0.3, kernel_num=16, feature_dim=512):
        super(AudioNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.feature_dim = feature_dim
        self.kernel_num  = kernel_num

        self.convt1 = nn.Conv2d(4, self.kernel_num, (3, 64))
        self.convt2 = nn.Conv2d(4, self.kernel_num, (5, 64))
        self.convt3 = nn.Conv2d(4, 16, (10, 64))
        self.convf1 = nn.Conv2d(4, self.kernel_num, (64, 3))
        self.convf2 = nn.Conv2d(4, self.kernel_num ,(64, 5))
        self.convf3 = nn.Conv2d(4, 16, (64, 10))

        self.fc_audio = nn.Linear(244*self.kernel_num, self.feature_dim)

        self.dropout1 = nn.Dropout2d(self.dropout_rate)
        self.dropout2 = nn.Dropout2d(self.dropout_rate)
        self.dropout3 = nn.Dropout2d(self.dropout_rate)
        self.dropout4 = nn.Dropout2d(self.dropout_rate)
        self.bn1 = nn.BatchNorm2d(self.kernel_num)
        self.bn2 = nn.BatchNorm2d(self.kernel_num)
        self.bn3 = nn.BatchNorm2d(self.kernel_num)
        self.bn4 = nn.BatchNorm2d(self.kernel_num)

    def forward(self, x):
        [b, c, row, col] = x.size()

        t1 = F.relu(self.convt1(x))
        # t1 = self.bn1(t1)
        t1 = self.dropout1(t1).view(b, -1)

        t2 = F.relu(self.convt2(x))
        # t2 = self.bn2(t2)
        t2 = self.dropout2(t2).view(b, -1)

        f1 = F.relu(self.convf1(x))
        # f1 = self.bn3(f1)
        f1 = self.dropout3(f1).view(b, -1)

        f2 = F.relu(self.convf2(x))
        # f2 = self.bn4(f2)
        f2 = self.dropout4(f2).view(b, -1)

        feature_tf = torch.cat([t1, t2, f1, f2], dim=-1)
        feature_tf = feature_tf.view(b, 244*self.kernel_num)
        feature_tf_audio = F.relu(self.fc_audio(feature_tf))
        feature_tf = feature_tf_audio.view(b, 1, self.feature_dim)

        return feature_tf

