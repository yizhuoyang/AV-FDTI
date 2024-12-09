import torch.nn as nn
import torch.nn.functional as F
import torch
from nets.attention import attentionLayer
from nets.audio_net import AudioNet
from nets.visual_net import VisualNet
from nets.detection_heads_hm import DetectNet


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

