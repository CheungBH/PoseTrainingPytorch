import torch
import torch.nn as nn
from torch.autograd import Variable

from .SE_Resnet18 import SEResnet
from models.duc.DUC import DUC
from src.opt import opt


class SeResPose18(nn.Module):
    DIM = 32

    def __init__(self, cfg_file):
        super(SeResPose18, self).__init__()

        cfg = None
        head_inp, duc1_out, duc2_out = 512, 256, 128
        if cfg_file:
            with open(cfg_file) as file:
                data = file.readlines()
            cfg = data[0].replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").split(",")
            cfg = [int(i) for i in cfg]
            if len(cfg) > 10:   # shortcut pruning
                head_inp, duc1_out, duc2_out = cfg[-3], cfg[-2], cfg[-1]

        self.seresnet18 = SEResnet(cfg=cfg)

        self.shuffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(int(head_inp/4), duc1_out, upscale_factor=2)
        self.duc2 = DUC(int(duc1_out/4), duc2_out, upscale_factor=2)

        self.conv_out = nn.Conv2d(int(duc2_out/4), opt.kps, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Variable):
        out = self.seresnet18(x)
        out = self.shuffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)

        return out


def createModel(cfg=None):
    return SeResPose18(cfg)


def test():
    net = createModel()
    y = net(torch.randn(1,3,64,64))
    print(net, file=open("FastPose.txt","w"))
    print(y.size())


if __name__ == '__main__':
    test()

