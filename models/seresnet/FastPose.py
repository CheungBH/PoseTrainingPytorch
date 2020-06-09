import torch
import torch.nn as nn
from torch.autograd import Variable

from .layers.SE_Resnet import SEResnet
from models.duc.DUC import DUC
from src.opt import opt
from config.config import train_body_part


class FastPose(nn.Module):
    DIM = 128

    def __init__(self, cfg_file):
        super(FastPose, self).__init__()

        cfg = None
        if cfg_file:
            with open(cfg_file) as file:
                data = file.readlines()
            cfg = data[0].replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").split(",")
            cfg = [int(i) for i in cfg]

        self.preact = SEResnet(cfg=cfg)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)

        if "duc" in opt.loadModel:
            self.conv_out = nn.Conv2d(self.DIM, 33, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_out = nn.Conv2d(
                self.DIM, len(train_body_part), kernel_size=3, stride=1, padding=1)

    def forward(self, x: Variable):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        return out


def createModel(cfg=None):
    return FastPose(cfg)


def test():
    net = createModel()
    y = net(torch.randn(1,3,64,64))
    print(net, file=open("FastPose.txt","w"))
    print(y.size())


if __name__ == '__main__':
    test()

