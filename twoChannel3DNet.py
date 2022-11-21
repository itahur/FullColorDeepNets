import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Network in which the input is 3D and each channel is evaluated separately
class twoChannel3DNet(nn.Module):
    
    def __init__(self, layers, output_neurons, features, mix, num_classes=10, init_weights=True, dropout=0.5):
        super(twoChannel3DNet, self).__init__()
        
        self.features = self.make_layers(layers, mix)
        self.avgpool = nn.AdaptiveAvgPool3d((features[0], features[1], features[2]))
        self.classifier = nn.Sequential(nn.Linear(output_neurons * features[0] * features[1] * features[2] , num_classes))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_inputs = torch.cat([x, x[:, :, 0, :, :].unsqueeze(2)], dim=2)
        new_inputs = new_inputs.cuda()
        x = self.features(new_inputs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # Create layers for the network
    def make_layers(self, cfg, mix, batch_norm = False):
        layers: List[nn.Module] = []
        in_channels = 1
        for out_channels in cfg:
            if out_channels == "M":
                layers += [nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))]
            else:
                out_channels = int(out_channels)         
                
                if mix and in_channels > 1:
                    conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
                    layers += [conv3d, nn.ReLU(inplace=True)]
                else:
                    if in_channels == 1:
                        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1))
                    else:
                        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                    if batch_norm:
                        layers += [conv3d, nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv3d, nn.ReLU(inplace=True)]
                in_channels = out_channels
        return nn.Sequential(*layers)