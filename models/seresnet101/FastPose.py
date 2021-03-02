import torch
import torch.nn as nn
from torch.autograd import Variable

from .SE_Resnet101 import SEResnet
from models.duc.DUC import DUC
from src.opt import opt
from config.config import train_body_part, device


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

        self.conv_out = nn.Conv2d(self.DIM, opt.kps, kernel_size=3, stride=1, padding=1)

        if opt.loadModel:
            if "duc" in opt.loadModel:
                self.conv_out = nn.Conv2d(self.DIM, 33, kernel_size=3, stride=1, padding=1)


    def forward(self, x: Variable):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        if opt.loadModel:
            if "duc" in opt.loadModel:
                out = out.narrow(1, 0, 17)

        return out


def createModel(cfg=None):
    return FastPose(cfg)


class InferenNet_fast(nn.Module):
    def __init__(self, dataset="coco", cfg=None):
        super(InferenNet_fast, self).__init__()
        if device != "cpu":
            model = createModel(cfg=cfg).cuda()
        else:
            model = createModel(cfg=cfg)
        print('Loading pose model from {}'.format(opt.loadModel))
        model.load_state_dict(torch.load(opt.loadModel, map_location=device))

        model.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        if opt.loadModel:
            if "duc" in opt.loadModel:
                out = out.narrow(1, 0, 17)

        return out


def test():
    net = createModel()
    y = net(torch.randn(1,3,64,64))
    print(net, file=open("FastPose.txt","w"))
    print(y.size())


if __name__ == '__main__':
    test()

