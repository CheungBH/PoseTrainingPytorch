import torch
import torch.nn as nn
import torch.nn.functional as F
from models.duc.DUC import DUC
# from efficientnet_pytorch import EfficientNet
from models.efficientnet.altered_efficientnet import EfficientNet
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

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
        #                        kernel_size=3,
        #                        stride=1,padding=1)
        # self.conv1_bn = nn.BatchNorm2d(32)


        self.efficient_highres = EfficientNet.from_name('efficientnet-b4-epII')
        self.efficient_lowres = EfficientNet.from_name('efficientnet-b0-epII')

        self.shuffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(int(duc1/2), duc1, upscale_factor=2)
        self.duc2 = DUC(int(duc1/4), duc2, upscale_factor=2)
        #self.duc3 = DUC(128, 256, upscale_factor=2)
        #self.duc4 = DUC(256, 512, upscale_factor=2)
        self.conv_out = nn.Conv2d(
            int(duc2/4), n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x = F.relu(self.conv1_bn(self.conv1(x)))
        amp_resize = nn.AdaptiveAvgPool2d((180,180))
        x_highres = x
        x_lowres = amp_resize(x)
        out1 = self.efficient_highres(x_highres)
        out2 = self.efficient_lowres(x_lowres)

        out = torch.cat((out1, out2), 1)
        #out = self.shuffle1(out)
        #out = self.duc1(out)
        #out = self.duc2(out)
        #out = self.duc3(out)
        #out = self.duc4(out)

        #out = self.conv_out(out)
        return out

if __name__ == '__main__':
     model = createModel()
     inp = Variable(torch.rand(1, 3, 360, 360), requires_grad = True)
     out = model(inp)