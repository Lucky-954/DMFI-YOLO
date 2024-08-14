import torch
import torch.nn as nn
from einops import rearrange
from ..modules.conv import Conv, DWConv, RepConv, autopad
from .deformConv import DeformConv2d

__all__ = ['RFDCA']

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SE(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(SE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(ratio, in_channel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            b, c= x.shape[0:2]
            y = self.gap(x).view(b, c)
            y = self.fc(y).view(b, c,1, 1)
            return y

class RFDCA(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, reduction=32):
        super(RFDCA, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp,inp * (kernel_size**2),kernel_size,padding=kernel_size//2,
                                                stride=stride,groups=inp,
                                                bias =False),
                                      nn.BatchNorm2d(inp * (kernel_size**2)),
                                      nn.ReLU()
                                      )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = DeformConv2d(mip, oup, 3, padding=1, bias=False)
        self.conv_w = DeformConv2d(mip, oup, 3, padding=1, bias=False)
        # self.conv_h = SCConv(mip, oup, g=1)
        # self.conv_w = SCConv(mip, oup, g=1)
        self.conv = nn.Sequential(nn.Conv2d(inp,oup,kernel_size,stride=kernel_size))
        

    def forward(self, x):
        b,c = x.shape[0:2]
        generate_feature = self.generate(x)
        h,w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b,c,self.kernel_size**2,h,w)
        
        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        
        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        h,w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return self.conv(generate_feature * a_w * a_h)