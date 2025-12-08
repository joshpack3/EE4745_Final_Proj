import torch
import torch.nn as nn

# --- Model 1: Custom CNN (from Problem A) ---

class CustomCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    """
    def __init__(self, num_classes, input_channels=3):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Block 3 (Target for GradCAM in Problem A)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )
        # Target layer name for GradCAM (Problem A)
        self.target_layer_name = 'features.8' 

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

# --- Model 2: Lightweight ResNet Block Definition (from Problem A) ---

class BasicBlock(nn.Module):
    """Basic block used in small ResNet architectures."""
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class ResNetSmall(nn.Module):
    """A lightweight ResNet-style network."""
    def __init__(self, num_classes):
        super(ResNetSmall, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        def _make_layer(in_planes, planes, stride, num_blocks):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            current_in_planes = in_planes
            for stride in strides:
                layers.append(BasicBlock(current_in_planes, planes, stride))
                current_in_planes = planes
            return nn.Sequential(*layers)
        
        self.layer1 = _make_layer(16, 16, stride=1, num_blocks=2) 
        self.layer2 = _make_layer(16, 32, stride=2, num_blocks=2) 
        self.layer3 = _make_layer(32, 64, stride=2, num_blocks=2) 
        self.layer4 = _make_layer(64, 128, stride=2, num_blocks=2) 
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.linear = nn.Linear(128, num_classes)

        # Target layer name for GradCAM (Problem A)
        self.target_layer_name = 'layer4.1.bn2' 

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out