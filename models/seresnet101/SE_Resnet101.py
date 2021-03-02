import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SE.SE_module import SELayer
from src.opt import opt

se_ratio = opt.se_ratio


class SeBottleneck(nn.Module):
    """ SeBottleneck """
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, reduction=False):
        super(SeBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv3 = nn.Conv2d(cfg[1], planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        if reduction:
            self.se = SELayer(planes * 4, reduction=se_ratio)

        self.reduc = reduction
        if se_ratio < 0:
            self.reduc = False
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class SeResnet(nn.Module):
    """ SeResnet """

    def __init__(self,  block, num_blocks, cfg=None):
        super(SeResnet, self).__init__()

        self.inplanes = 64
        self.block = block

        if cfg is None:
            # Construct config variable.
            # only purned layer
            cfg = [[64, 64], [64, 64]*2, 
                   [128, 128], [128, 128]*3, 
                   [256, 256], [256, 256]*22, 
                   [512, 512], [512, 512]*2 ]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], 
                                  cfg = cfg[0:2*sum(num_blocks[0:1])], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], 
                                  cfg = cfg[2*sum(num_blocks[0:1]):2*sum(num_blocks[0:2])], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], 
                                  cfg = cfg[2*sum(num_blocks[0:2]):2*sum(num_blocks[0:3])], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], 
                                  cfg = cfg[2*sum(num_blocks[0:3]):2*sum(num_blocks[0:4])], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample, reduction=True))
        else:
            layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[2*i: 2*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        # x = self.relu(self.bn0(self.conv0(x))) # 64 * h/2 * w/2
        x = self.layer1(x)  # 256 * h/2 * w/2
        x = self.layer2(x)  # 512 * h/4 * w/4
        x = self.layer3(x)  # 1024 * h/8 * w/8
        x = self.layer4(x)  # 2048 * h/16 * w/16

        # x = F.avg_pool2d(x, 2)
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        return x


def SEResnet(cfg=None): # with maxpooling
    return SeResnet(SeBottleneck, [3,4,23,3], cfg)


def test():
    net = SEResnet()
    y = net(torch.randn(1,3,64,64))
    print(net, file=open("SEResnet.txt","w"))
    print(y.size())


if __name__ == '__main__':
    test()
