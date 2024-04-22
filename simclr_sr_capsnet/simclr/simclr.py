import torch.nn as nn
import torchvision
from simclr.modules.identity import Identity


class SimCLR(nn.Module):
    def __init__(self, encoder, caps_net):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.caps_net = caps_net

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()


    def forward(self, x_i):     
        x_i = self.encoder.conv1(x_i)
        x_i = self.encoder.bn1(x_i)
        x_i = self.encoder.relu(x_i)
        x_i = self.encoder.maxpool(x_i)
        x_i = self.encoder.layer1(x_i)
        x_i = self.encoder.layer2(x_i)
        x_i = self.encoder.layer3(x_i)
        
        h_i = self.encoder.layer4(x_i)
        # Output after layer4. We take the fm from the backbone [64, 2048, 7, 7])
        # and pass it through the capsule network
        z_i = self.caps_net(h_i) 
        return z_i