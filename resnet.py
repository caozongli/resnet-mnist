import torch
import torch.nn.modules as nn
from torch.nn import init


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1  # 输入到输出图像通道数有没有变化或者变化了多少倍

    def __init__(self, in_planes=None, planes=None, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            res = self.downsample(res)
        x += res
        x = self.relu2(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplane, plane, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplane, plane, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(plane)
        self.conv2 = nn.Conv2d(plane, plane, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(plane)
        self.conv3 = nn.Conv2d(plane, self.expansion * plane, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * plane)
        self.downsample = downsample
        self.stride = stride
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            res = self.downsample(res)
        x += res
        x = self.relu3(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.initdata = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layers(block, 64, layers[0])
        self.layer2 = self.make_layers(block, 128, layers[1], stride=1)
        self.layer3 = self.make_layers(block, 256, layers[2], stride=1)
        self.layer4 = self.make_layers(block, 512, layers[3], stride=2)

        self.avg = nn.AvgPool2d(7, stride=1)
        self.full = nn.Linear(512 * block.expansion, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):    # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


    def make_layers(self, block, planes, num_blocks, stride=1):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != block.expansion * planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, block.expansion * planes, kernel_size=1,
                                                 stride=stride, bias=False),
                                       nn.BatchNorm2d(block.expansion * planes))
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initdata(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.full(x)
        x = self.sigmoid(x)

        return x



def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])



