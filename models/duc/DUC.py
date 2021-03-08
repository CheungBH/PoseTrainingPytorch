import torch.nn as nn
from ..utils.utils import parse_DUC


class DUC(nn.Module):
    '''
    INPUT: inplanes, planes, upscale_factor
    OUTPUT: (planes // 4)* ht * wd
    '''
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class DUC_head(nn.Module):
    def __init__(self, cfg, head_inp, upscale_factor=2):
        super(DUC_head, self).__init__()
        self.shuffle = nn.PixelShuffle(upscale_factor)
        self.conv_size, self.kps = parse_DUC(cfg)
        self.head_inp = head_inp
        self.upscale_factor = upscale_factor
        self.DUC_layers = self.build_DUC_head()
        self.head = nn.Conv2d(int(self.conv_size[-1]/4), self.kps, kernel_size=3, stride=1, padding=1)

    def build_DUC_head(self):
        DUC_layers = [DUC(int(self.head_inp/4), self.conv_size[0], upscale_factor=self.upscale_factor)]
        for idx in range(len(self.conv_size))[:-1]:
            DUC_layers.append(DUC(int(self.conv_size[idx]/4), self.conv_size[idx+1], upscale_factor=self.upscale_factor))
        return nn.Sequential(*DUC_layers)

    def forward(self, x):
        x = self.shuffle(x)
        x = self.DUC_layers(x)
        x = self.head(x)
        return x


