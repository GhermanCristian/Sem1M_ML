import torch.nn as nn
from unit import Unit


class SimpleNet(nn.Module):
    def __init__(self, classCount):
        super(SimpleNet, self).__init__()

        # 14 layers, max pooling in between
        # RGB => 3 input channels
        self.unit1 = Unit(3, 32)
        self.unit2 = Unit(32, 32)
        self.unit3 = Unit(32, 32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(32, 64)
        self.unit5 = Unit(64, 64)
        self.unit6 = Unit(64, 64)
        self.unit7 = Unit(64, 64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(64, 128)
        self.unit9 = Unit(128, 128)
        self.unit10 = Unit(128, 128)
        self.unit11 = Unit(128, 128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(128, 128)
        self.unit13 = Unit(128, 128)
        self.unit14 = Unit(128, 128)

        self.avgPool = nn.AvgPool2d(kernel_size=4)

        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 , self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgPool)

        self.fc = nn.Linear(in_features=128 * 7 * 7, out_features=classCount)

    def forward(self, inputData):
        return self.fc(self.net(inputData).view(-1, 128 * 7 * 7))
