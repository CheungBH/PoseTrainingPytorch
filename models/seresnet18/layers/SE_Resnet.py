import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seresnet.layers.SE_module import SELayer


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SeBasicBlock(nn.Module):
    """ SeBottleneck """
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, reduction=False):
        super(SeBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(cfg, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        if reduction:
            self.se = SELayer(planes)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SeResnet(nn.Module):
    """ SeResnet """

    def __init__(self,  block, num_blocks, cfg=None):
        super(SeResnet, self).__init__()

        self.inplanes = 64
        self.block = block

        if not cfg:
            # Construct config variable.
            # only purned layer
            cfg = [[64, 64], [128, 128], [256, 256], [512, 512]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], cfg=cfg[0:2], stride=1, se=True)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], cfg=cfg[2:4], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], cfg=cfg[4:6], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], cfg=cfg[6:8], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, planes, blocks, cfg, stride=1, se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes, cfg[0], stride, downsample, reduction=True))
        else:
            if se:
                layers.append(block(self.inplanes, planes, cfg[0], stride, downsample, reduction=True))
            else:
                layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))

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
    return SeResnet(SeBasicBlock, [2,2,2,2], cfg)


def test():
    net = SEResnet()
    y = net(torch.randn(1,3,64,64))
    print(net, file=open("SEResnet.txt","w"))
    print(y.size())


if __name__ == '__main__':
    test()
