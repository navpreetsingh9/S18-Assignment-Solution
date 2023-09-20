import torch
import torch.nn as nn

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strconv):
        super(ContractingBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        if strconv:
            self.maxpool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x, mp=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        skip = x  # store the output for the skip connection
        if mp:
            x = self.maxpool(x)
        
        return x, skip

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False):
        super(ExpandingBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        if up:
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                          nn.Conv2d(in_channels, in_channels // 2, kernel_size=1))

        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat((x, skip), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, strconv=False, up=False):
        super(UNet, self).__init__()
        
        self.contract1 = ContractingBlock(in_channels, 64, strconv)
        self.contract2 = ContractingBlock(64, 128, strconv)
        self.contract3 = ContractingBlock(128, 256, strconv)
        self.contract4 = ContractingBlock(256, 512, strconv)
        
        self.expand1 = ExpandingBlock(512, 256, up)
        self.expand2 = ExpandingBlock(256, 128, up)
        self.expand3 = ExpandingBlock(128, 64, up)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Contracting path
        x1, skip1 = self.contract1(x)
        x2, skip2 = self.contract2(x1)
        x3, skip3 = self.contract3(x2)
        x4, _ = self.contract4(x3, mp=False)
        
        # Expanding path
        x5 = self.expand1(x4, skip3)
        x6 = self.expand2(x5, skip2)
        x7 = self.expand3(x6, skip1)

        return self.final_conv(x7)
