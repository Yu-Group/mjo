import torch
from torch import nn
from torch.nn import functional as F

class UNet(nn.Module):

    """
    Args:
        nc (int): number of output channels for first 
    """
    def __init__(self, nc=64, nc_penum=128, nc_final=128, kernel_size=5, stride=1, padding='same'):
        super().__init__()
        self.conv1 = nn.Conv2d(2, nc, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(nc, nc, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(nc, nc, kernel_size, stride, padding)
        self.conv4 = nn.Conv2d(nc, nc, kernel_size, stride, padding)
        self.conv5 = nn.Conv2d(nc, nc, kernel_size, stride, padding)
        self.conv6 = nn.Conv2d(2*nc, 2*nc, kernel_size, stride, padding)
        self.conv7 = nn.Conv2d(3*nc, 2, kernel_size, stride, padding)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = torch.cat((F.relu(self.conv5(x4)), x3), dim=1)
        x6 = torch.cat((F.relu(self.conv6(x5)), x2), dim=1)
        out = self.conv7(x6)
        return out, x1, x2, x3, x4, x5, x6
