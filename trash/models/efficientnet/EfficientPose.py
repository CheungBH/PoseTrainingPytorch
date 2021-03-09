import torch
import torch.nn as nn
from models.duc.DUC import DUC
# from efficientnet_pytorch import EfficientNet
from models.efficientnet.efficientnet import EfficientNet
from config.config import train_body_part, DUCs

from torch.autograd import Variable

n_classes = len(train_body_part)

duc1 = DUCs[0]
duc2 = DUCs[1]


def createModel(cfg='efficientnet-b0'):
    return EfficientPose(cfg)


class EfficientPose(nn.Module):
    def __init__(self, cfg):
        super(EfficientPose, self).__init__()

        self.efficient = EfficientNet.from_name(cfg)

        self.shuffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(int(duc1/2), duc1, upscale_factor=2)
        self.duc2 = DUC(int(duc1/4), duc2, upscale_factor=2)
        #self.duc3 = DUC(128, 256, upscale_factor=2)
        #self.duc4 = DUC(256, 512, upscale_factor=2)
        self.conv_out = nn.Conv2d(
            int(duc2/4), n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.efficient(x)
        out = self.shuffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)
        #out = self.duc3(out)
        #out = self.duc4(out)

        out = self.conv_out(out)
        return out

if __name__ == '__main__':
     model = createModel()
     inp = Variable(torch.rand(1, 3, 360, 360), requires_grad = True)
     out = model(inp)