import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Small 3D convolutional network based on the VGGNet
class vggm3DNet(nn.Module):
    
    def __init__(self, layers, num_classes=8, init_weights=True, dropout=0.5):
        super(vggm3DNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, layers[0], kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(layers[0], layers[1], kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(layers[1], layers[2], kernel_size=3, stride=1, padding=1),
            nn.Conv3d(layers[2], layers[3], kernel_size=3, stride=1, padding=1),
            nn.Conv3d(layers[3], layers[4], kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
        )
        self.avgpool = nn.AdaptiveAvgPool3d((3, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(layers[4] * 3, 4096),
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