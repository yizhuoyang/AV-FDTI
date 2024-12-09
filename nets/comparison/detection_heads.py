import torch.nn as nn
import torch.nn.functional as F

"""
This network is the detection head which used for the comparison methods. The detection deads provides the 3D location, class and trajactory (T*3) of the target drone.
"""

class DetectNet(nn.Module):
    def __init__(self, feature_dim=512, num_class=6):
        super(DetectNet, self).__init__()
        self.feature_dim = feature_dim
        self.num_class   = num_class
        #  position heads
        self.position_est1 = nn.Linear(self.feature_dim, 256)
        self.position_est2 = nn.Linear(256, 128)
        self.position_est3 = nn.Linear(128, 3)

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
        class_detection =F.relu( self.cls2(class_detection))
        class_detection = self.cls3(class_detection)

        # trajectory detection
        trajectory = F.relu(self.traj1(f_all))
        trajectory  = F.relu(self.traj2(trajectory))
        trajectory  = self.traj3(trajectory)
        trajectory  = trajectory .view(-1, 10, 3)

        position = F.relu(self.position_est1(f_all))
        position  = F.relu(self.position_est2(position))
        position  = self.position_est3(position)
        return position,class_detection,trajectory



