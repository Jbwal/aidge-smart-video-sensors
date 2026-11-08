import torch
import torch.nn as nn

class PersonDetectorCNN(nn.Module):

    #filtres utilisés par le réseau de neurone
    def __init__(self, num_classes=1):
        super(PersonDetectorCNN, self).__init__()
        
        #bloc 1 : 160x120 -> 80x60
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        #bloc 2 : 80x60 -> 40x30 (depthwise separable vonv pour save la memoire)
        self.dw_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16)
        self.pw_conv2 = nn.Conv2d(16, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        #bloc 3 : 40x30 -> 20x15
        self.dw_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32)
        self.pw_conv3 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        #bloc 4 : poooling global 20x15 -> 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        #classification : on choisis si on detecte une personne ou pas
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.pw_conv2(self.dw_conv2(x))))
        x = self.relu3(self.bn3(self.pw_conv3(self.dw_conv3(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = PersonDetectorCNN()
    dummy_input = torch.randn(1, 1, 120, 160)
    output = model(dummy_input)
    print("modele ok")
    print(f"{output.shape}")