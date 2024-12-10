import torch.nn as nn
import torch.nn.functional as F
import torch
from nets.attention import attentionLayer
from nets.audio_net import AudioNet
from nets.visual_net import VisualNet
from nets.detection_heads_hm import DetectNet
import torch.nn as nn
import torch.nn.functional as F
import torch

class AVFDTI(nn.Module):
    def __init__(self, dropout_rate=0.3, kernel_num=32, feature_dim=512,num_class=6):
        super(AVFDTI,self).__init__()
        self.feature_dim = feature_dim
        # Visual Def
        self.visualnet = VisualNet(feature_dim=feature_dim)
        # Audio Def
        self.audionet  = AudioNet(dropout_rate,kernel_num,feature_dim)
        # Attention module
        self.attention = attentionLayer(feature_dim,8)
        # Detection heads
        self.detection = DetectNet(feature_dim,kernel_num=kernel_num,num_class=num_class)
        # drone detection module
        self.fc_detect = nn.Linear(feature_dim, 256)
        self.fc_detect2 = nn.Linear(256, 128)
        self.fc_detect3 = nn.Linear(128, 2)
    def forward(self, x, y):
        [b, c, row, col] = x.size()
        # Audio feature extraction
        feature_tf    = self.audionet(x)
        # Visual feature extraction
        feature_vis   = self.visualnet(y)
        f_tem = feature_vis.view(-1, 1, self.feature_dim)

        # Feature Fusion
        f_all = torch.cat([feature_tf, f_tem], dim=1)
        detect = F.relu(self.fc_detect(feature_vis))
        detect = F.relu(self.fc_detect2(detect))
        detect = self.fc_detect3(detect)
        detect_soft = F.softmax(detect,dim=-1)
        detect_expanded = detect_soft.unsqueeze(-1)
        f_all = f_all * detect_expanded

        f_all = torch.sum(f_all, 1)
        audio_feature = feature_tf.float()
        f_all = self.attention(f_all.view(b,1,-1),audio_feature)
        f_all = f_all.view(b,-1)
        #Detection module
        x, o, p, c,t= self.detection(f_all)
        return x, p, o, detect, t, c
#
# class AVFDTI(nn.Module):
#     def __init__(self, dropout_rate=0.3, kernel_num=32, feature_dim=512,num_class=6):
#         super().__init__()
#         # Visual Def
#         self.feature_dim = feature_dim
#         self.resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#         self.resnet_feature = nn.Sequential(*list(self.resnet_model.children())[:-2])
#         self.convvis = nn.Conv2d(256, kernel_num, (3, 3), padding=1)
#         self.convres = nn.Conv2d(2048, 256, (3, 3), padding=1)
#         self.fcvis = nn.Linear(2048, feature_dim)
#
#         # Heatmap conv and deconv
#         self.deconv1 = nn.ConvTranspose2d(kernel_num, kernel_num, 3, stride=2, padding=1, output_padding=1)
#         self.deconv2 = nn.ConvTranspose2d(kernel_num, kernel_num, 3, stride=2, padding=1, output_padding=1)
#         self.deconv3 = nn.ConvTranspose2d(kernel_num, kernel_num, 3, stride=2, padding=1, output_padding=1)
#         self.deconv4 = nn.ConvTranspose2d(kernel_num, kernel_num, 3, stride=2, padding=1, output_padding=1)
#         self.deconv0 = nn.ConvTranspose2d(1, kernel_num, 3, stride=2, padding=1, output_padding=1)
#         self.conv2 = nn.Conv2d(kernel_num, kernel_num, (3, 3), padding=1)
#         self.conv3 = nn.Conv2d(kernel_num, kernel_num, (3, 3), padding=1)
#         self.conv4 = nn.Conv2d(kernel_num, kernel_num, (3, 3), padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.bn3 = nn.BatchNorm2d(16)
#         self.bn4 = nn.BatchNorm2d(16)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.final_conv = nn.Conv2d(kernel_num, 1, (3, 3), padding=1)
#
#         # Audio Def
#         self.convt1 = nn.Conv2d(4, 16, (3, 64))
#         self.convt2 = nn.Conv2d(4, 16, (5, 64))
#         self.convt3 = nn.Conv2d(4, 16, (10, 64))
#         self.convf1 = nn.Conv2d(4, 16, (64, 3))
#         self.convf2 = nn.Conv2d(4, 16, (64, 5))
#         self.convf3 = nn.Conv2d(4, 16, (64, 10))
#         self.fc1 = nn.Linear(1024, feature_dim)
#         self.dropout1 = nn.Dropout2d(dropout_rate)
#         self.dropout2 = nn.Dropout2d(dropout_rate)
#         self.dropout3 = nn.Dropout2d(dropout_rate)
#         self.dropout4 = nn.Dropout2d(dropout_rate)
#         self.lstm1 = nn.LSTM(input_size=3904, hidden_size=2048, num_layers=1, batch_first=True, bidirectional=False)
#         self.lstm2 = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=1, batch_first=True, bidirectional=False)
#         self.attention3 = attentionLayer(self.feature_dim,8)
#         # Detection heads
#         self.fc2 = nn.Linear(feature_dim, 256)
#         self.fc3 = nn.Linear(256, 100)
#         self.fc4 = nn.Linear(100, 1)
#
#         self.detect = nn.Linear(feature_dim, 256)
#         self.detec2 = nn.Linear(256, 128)
#         self.detec3 = nn.Linear(128, 2)
#
#         self.offset2 = nn.Linear(feature_dim, 256)
#         self.offset3 = nn.Linear(256, 100)
#         self.offset4 = nn.Linear(100, 2)
#
#         self.traj2 = nn.Linear(512, 256)
#         self.traj3 = nn.Linear(256, 100)
#         self.traj4 = nn.Linear(100, 30)
#
#         self.cls2 = nn.Linear(feature_dim, 256)
#         self.cls3 = nn.Linear(256, 100)
#         self.cls4 = nn.Linear(100, 6)
#
#         self.fc_all = nn.Linear(feature_dim, 256)
#
#         self.dropout_conv1 = nn.Dropout2d(0.3)
#         self.dropout_conv2 = nn.Dropout2d(0.3)
#
#         self.dropout_deconv1 = nn.Dropout2d(0.3)
#         self.dropout_deconv2 = nn.Dropout2d(0.3)
#         self.dropout_deconv3 = nn.Dropout2d(0.3)
#         self.dropout_deconv4 = nn.Dropout2d(0.3)
#
#     def forward(self, x, y):
#         [b, c, row, col] = x.size()
#         # AudioNet
#         t1 = F.relu(self.convt1(x))
#         # t1 = self.bn1(t1)
#         t1 = self.dropout1(t1).view(b, -1)
#
#         t2 = F.relu(self.convt2(x))
#         # t2 = self.bn2(t2)
#         t2 = self.dropout2(t2).view(b, -1)
#
#         f1 = F.relu(self.convf1(x))
#         # f1 = self.bn3(f1)
#         f1 = self.dropout3(f1).view(b, -1)
#
#         f2 = F.relu(self.convf2(x))
#         # f2 = self.bn4(f2)
#         f2 = self.dropout4(f2).view(b, -1)
#
#         feature_tf = torch.cat([t1, t2, f1, f2], dim=-1)
#         feature_tf = feature_tf.view(b, -1, 3904)
#         feature_tf, _ = self.lstm1(feature_tf)
#         feature_tf, _ = self.lstm2(feature_tf)
#         feature_tf = feature_tf.view(b, -1)
#         feature_tf_audio = F.relu(self.fc1(feature_tf))
#         feature_tf = feature_tf_audio.view(b, 1, self.feature_dim)
#
#         # VisualNet
#         y = self.resnet_feature(y)
#         y = F.relu(self.convres(y))
#         y = F.relu(self.convvis(y))
#         fi = y.view(-1, 2048)
#         fi = F.relu(self.fcvis(fi))
#         fi_tem = fi.view(b, 1, self.feature_dim)
#         # print(fi_tem.shape,feature_tf.shape)
#         # Feature Fusion
#         f_all = torch.cat([feature_tf, fi_tem], dim=1)
#         detect = F.relu(self.detect(fi))
#         detect = F.relu(self.detec2(detect))
#         detect = self.detec3(detect)
#         detect_soft = F.softmax(detect)
#         detect_expanded = detect_soft.unsqueeze(-1)
#         f_all = f_all * detect_expanded
#         f_all = torch.sum(f_all, 1)
#
#
#         audio_feature = feature_tf.float()
#         f_all = self.attention3(f_all.view(b,1,-1),audio_feature)
#
#
#         f_all = f_all.view(b,-1)
#         ##################  Detection module #####################
#         # Predict Trajectoy
#         t = F.relu(self.traj2(feature_tf_audio))
#         t = F.relu(self.traj3(t))
#         t = self.traj4(t)
#         t = t.view(-1, 10, 3)
#
#         # Predict class
#
#         c = F.relu(self.cls2(f_all))
#         c = F.relu(self.cls3(c))
#         c = self.cls4(c)
#
#         # Predict z axis
#         p = F.relu(self.fc2(f_all))
#         p = F.relu(self.fc3(p))
#         p = self.fc4(p)
#
#         # Predict offset
#         o = F.relu(self.offset2(f_all))
#         o = F.relu(self.offset3(o))
#         o = self.offset4(o)
#
#         # Predict heatmap
#         feature_tft = F.relu(self.fc_all(f_all))
#         feature_tft = feature_tft.view(-1, 1, 16, 16)
#         x = F.relu(self.deconv0(feature_tft))
#
#         x = F.relu(self.deconv1(x))
#         # x = self.dropout_deconv1(x)
#         x = F.relu(self.deconv2(x))
#         # x = self.dropout_deconv2(x)
#         x = F.relu(self.deconv3(x))
#         # x = self.dropout_deconv3(x)
#         x = F.relu(self.deconv4(x))
#         # x = self.dropout_deconv4(x)
#
#         x = F.relu(self.conv2(x))
#         # x = self.dropout_conv1(x)
#         x = F.sigmoid(self.final_conv(x))
#         x = x.view(-1, 512, 512)
#
#         return x, p, o, detect, t, c
