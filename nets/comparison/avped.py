import torch.nn as nn
import torch.nn.functional as F
import torch
from nets.audio_net import AudioNet
from nets.visual_net import VisualNet
from nets.comparison.detection_heads import DetectNet
from torchinfo import summary
"""
This network is the replementation of the paper: AV-PedAware: Self-Supervised Audio-Visual Fusion for Dynamic Pedestrian Awareness
"""

class AVPed(nn.Module):
    def __init__(self, feature_dim=256,classes=6):
        super(AVPed, self).__init__()
        self.feature_dim = feature_dim
        self.classes     = classes
        self.detectnet            = DetectNet(self.feature_dim,self.classes)
        self.audionet             = AudioNet(feature_dim=self.feature_dim)
        self.visualnet            = VisualNet(self.feature_dim)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout_att1 = nn.Dropout1d(0.4)


    def forward(self, x, y):
        [ba, ca, rowa, cola] = x.size()
        audio_feature        = self.audionet(x)
        visual_feature        = self.visualnet(y)
        visual_feature   = visual_feature.reshape(ba,1,-1)
        f_all = torch.cat([audio_feature, visual_feature], dim=1)
        fi = visual_feature.view(ba,-1)
        detect = F.relu(self.fc1(fi))
        detect = self.fc2(detect)
        detect_soft = F.softmax(detect,-1)
        detect_expanded = detect_soft.unsqueeze(-1)
        f_all = f_all * detect_expanded
        f_all = torch.sum(f_all, 1)
        position,class_detection,trajectory = self.detectnet(f_all)

        return position,detect,class_detection,trajectory

if __name__ == "__main__":

    feature_dim = 256
    classes = 6
    model = AVPed(feature_dim=feature_dim, classes=classes)

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define input sizes for audio (x) and visual (y) data
    input_shape_audio = (1, 4, 64, 64)  # Example for audio input
    input_shape_visual = (1, 3, 256,256)  # Example for visual input

    # Summarize the model using torchinfo
    print("AVPed Model Summary:")
    model_summary = summary(
        model,
        input_data=(torch.randn(*input_shape_audio).to(device), torch.randn(*input_shape_visual).to(device)),
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
    )

    dummy_audio = torch.randn(*input_shape_audio).to(device)
    dummy_visual = torch.randn(*input_shape_visual).to(device)

    # Forward pass
    position, detect, class_detection, trajectory = model(dummy_audio, dummy_visual)

    # Print output shapes
    print("\nForward Pass Results:")
    print("Position Output Shape:", position.shape)  # Expected: (batch_size, feature_dim)
    print("Detection Output Shape:", detect.shape)  # Expected: (batch_size, 2)
    print("Class Detection Output Shape:", class_detection.shape)  # Expected: (batch_size, classes)
    print("Trajectory Output Shape:", trajectory.shape)  # Depends on DetectNet implementation
