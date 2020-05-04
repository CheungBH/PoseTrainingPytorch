
import torch.nn as nn
from models.duc.DUC import DUC
from models.shufflenet.shufflenet import shufflenet_v2_x1_0
from config.config import train_body_part, DUCs

n_classes = len(train_body_part)


def createModel(cfg=None):
    return ShufflePose(cfg)


class ShufflePose(nn.Module):
    def __init__(self, setting):
        super(ShufflePose, self).__init__()
        self.shuffle = shufflenet_v2_x1_0()

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(320, DUCs[0], upscale_factor=2)
        self.duc2 = DUC(int(DUCs[0]/4), DUCs[1], upscale_factor=2)
        #self.duc3 = DUC(128, 256, upscale_factor=2)
        #self.duc4 = DUC(256, 512, upscale_factor=2)
        self.conv_out = nn.Conv2d(
            int(DUCs[1]/4), n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.shuffle(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)
        #out = self.duc3(out)
        #out = self.duc4(out)

        out = self.conv_out(out)
        return out
