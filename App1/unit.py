import torch.nn as nn


class Unit(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=inChannels, kernel_size=3, out_channels=outChannels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=outChannels)
        self.relu = nn.ReLU()

    def forward(self, inputData):
        return self.relu(self.bn(self.conv(inputData)))
