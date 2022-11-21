import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Small 2D convolutional network based on the VGGNet
class small2DNet(nn.Module):
    
    def __init__(self, layers, output_neurons, features, num_classes=10, init_weights=True, dropout=0.5):
        super(small2DNet, self).__init__()
        
        self.features = self.make_layers(layers)
        self.avgpool = nn.AdaptiveAvgPool2d((features[0], features[1]))
        self.classifier = nn.Sequential(
            nn.Linear(output_neurons * features[0] * features[1], num_classes),
            # nn.Linear(192, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(p=dropout),
            # nn.Linear(linear_neurons, linear_neurons),
            # nn.ReLU(True),
            # nn.Dropout(p=dropout),
            # nn.Linear(linear_neurons, num_classes),
        )
            
    # Forward function
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # Create layers for the network
    def make_layers(self, cfg, batch_norm = False):
        layers: List[nn.Module] = []
        in_channels = 3
        for out_channels in cfg:
            if out_channels == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = int(out_channels)
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = out_channels
        return nn.Sequential(*layers)