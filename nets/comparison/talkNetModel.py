import torch
import torch.nn as nn
from nets.comparison.talknet.audioEncoder      import audioEncoder
from  nets.comparison.talknet.visualEncoder     import visualFrontend, visualTCN, visualConv1D
from  nets.comparison.talknet.attentionLayer    import attentionLayer
import torch.nn.functional as F
from nets.comparison.detection_heads import DetectNet

"""
This script utilizes the network in paper: Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection. 
Reference: https://github.com/TaoRuijie/TalkNet-ASD
"""

class talkNetModel(nn.Module):
    def __init__(self):
        super(talkNetModel, self).__init__()
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend()
        self.visualTCN       = visualTCN()
        self.visualConv1D    = visualConv1D()
        self.detectnet            = DetectNet(256,6)
        # Audio Temporal Encoder 
        self.audioEncoder  = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128])
        self.crossA2V = attentionLayer(d_model = 128, nhead = 8)
        self.crossV2A = attentionLayer(d_model = 128, nhead = 8)
        self.selfAV = attentionLayer(d_model = 256, nhead = 8)
        self.fc   = nn.Linear(768,100)
        self.fc_c = nn.Linear(100,6)
        self.fc_p = nn.Linear(100,3)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x

    def forward_audio_frontend(self, x):
        x = self.audioEncoder(x)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src = x1, tar = x2)
        x2_c = self.crossV2A(src = x2, tar = x1)        
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2): 
        x = torch.cat((x1,x2), 2)    
        x = self.selfAV(src = x, tar = x)       
        x = torch.reshape(x, (-1, 256))
        return x    

    def forward_audio_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x


    def forward(self,x,y):
        b,c,h,w = x.size()
        audioEmbed = self.forward_audio_frontend(x) # feedForward
        visualEmbed = self.forward_visual_frontend(y)
        audioEmbed, visualEmbed = self.forward_cross_attention(audioEmbed, visualEmbed)

        outsAV= self.forward_audio_visual_backend(audioEmbed, visualEmbed)
        feature = F.relu(self.fc(outsAV.view(b,-1)))
        position = self.fc_p(feature)
        cls      = self.fc_c(feature)
        return position,cls



