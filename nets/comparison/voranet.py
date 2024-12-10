import torch.nn as nn
import torch
from torchinfo import summary
"""
This network is the replementation of the paper: DroneChase: A Mobile and Automated Cross-Modality System for Continuous Drone Tracking
"""

class voranet(nn.Module):
    def __init__(self,classes=6):
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
        self.fc3 = nn.Linear(32, classes)
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

if __name__ == "__main__":

    model = voranet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Define input size
    batch_size = 2
    input_shape = (batch_size, 4, 224,224)  # Example input: (batch_size, channels, height, width)
    #
    # # Summarize the model using torchinfo
    print("voranet Model Summary:")
    model_summary = summary(
        model,
        input_size=(1,4, 224,224),  # Exclude batch size
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
    )

    # Generate a sample to show the output shapes
    dummy_input = torch.randn(*input_shape).to(device)
    output_position, output_classification = model(dummy_input)

    # Print the output shapes
    print("\nForward Pass Results:")
    print("Position Output Shape:", output_position.shape)  # Expected: (batch_size, 3)
    print("Classification Output Shape:", output_classification.shape)  # Expected: (batch_size, 6)
