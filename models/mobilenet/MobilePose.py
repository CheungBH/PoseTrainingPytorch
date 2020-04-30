# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch.nn as nn
from models.duc.DUC import DUC
from models.mobilenet.mobilenet import MobileNetV2
from config.config import train_body_part

n_classes = len(train_body_part)


def createModel(cfg=None):
    return MobilePose(cfg)


class MobilePose(nn.Module):
    conv_dim = 80

    def __init__(self, setting):
        super(MobilePose, self).__init__()

        self.mobile = MobileNetV2(inverted_residual_setting=setting)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(320, 640, upscale_factor=2)
        self.duc2 = DUC(160, 320, upscale_factor=2)
        #self.duc3 = DUC(128, 256, upscale_factor=2)
        #self.duc4 = DUC(256, 512, upscale_factor=2)
        self.conv_out = nn.Conv2d(
            self.conv_dim, n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.mobile(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)
        #out = self.duc3(out)
        #out = self.duc4(out)

        out = self.conv_out(out)
        return out
