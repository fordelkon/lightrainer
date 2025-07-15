import torch
from torch import nn
from pytorch_quantization import nn as quant_nn
import math


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k] # actual kernel size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # auto-pad
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, 
                 g=1, d=1, act=True, has_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), 
                              groups=g, dilation=d, bias=has_bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        self.act(self.conv(x))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)
    

class ConvWithConstraint(nn.Module):
    def __init__(self, c1, c2, k=1, max_norm=1.0, 
                 s=1, p=None, g=1, d=1, act=True, has_bias=True):
        super().__init__()
        self.conv = Conv2dWithConstraint(c1, c2, k, max_norm=max_norm, stride=s, 
                                         padding=autopad(k, p, d), groups=g, dilation=d, 
                                         bias=has_bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    

class QuantConv2dWithConstraint(quant_nn.QuantConv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)
    

class DWConvWithConstraint(ConvWithConstraint):
    def __init__(self, c1, c2, max_norm=1.0, k=1, act=True, s=1, p=0, d=1):
        super().__init__(c1, c2, k, max_norm=max_norm, s=s, p=p, g=math.gcd(c1, c2), d=d, act=act)


class SeparableConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, 
                 d=1, act=True, has_bias=True):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, stride=s, padding=autopad(k, p, d), 
                            dilation=d, groups=c1, bias=has_bias)
        self.pw = nn.Conv2d(c1, c2, 1, stride=s, bias=has_bias)
        self.bn = nn.BatchNorm2d(c2)

        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))
    

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)
    

class QuantLinearWithConstraint(quant_nn.QuantLinear):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)