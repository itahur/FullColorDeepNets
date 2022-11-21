import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Small 2D convolutional network based on the VGGNet
class vggm2DNet(nn.Module):
    
    def __init__(self, layers, num_classes=8, init_weights=True, dropout=0.5):
        super(vggm2DNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, layers[0], kernel_size=7, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(layers[0], layers[1], kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(layers[1], layers[2], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(layers[2], layers[3], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(layers[3], layers[4], kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(layers[4], 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes),
        )
            
    # Forward function
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x