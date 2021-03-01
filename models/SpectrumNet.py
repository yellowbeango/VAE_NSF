"""SE-ResNet in PyTorch
Based on preact_resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SpectrumNet18', 'SpectrumNet50', 'SpectrumNet101']


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEPreActBlock(nn.Module):
    """SE pre-activation of the BasicBlock"""
    expansion = 1  # last_block_channel/first_block_channel

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(SEPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.se = SELayer(planes, reduction)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.leaky_relu(self.bn2(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class SEPreActBootleneck(nn.Module):
    """Pre-activation version of the bottleneck module"""
    expansion = 4  # last_block_channel/first_block_channel

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(SEPreActBootleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.se = SELayer(self.expansion * planes, reduction)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.leaky_relu(self.bn2(out)))
        out = self.conv3(F.leaky_relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class SpectrumNet(nn.Module):
    def __init__(self, block, num_blocks, input_channel=1, reduction=16, points=183):
        super(SpectrumNet, self).__init__()
        self.points = points
        self.in_planes = 64
        self.convbnrelu = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2, reduction=reduction)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, reduction=reduction)
        self.conv_spectrum = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 512,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.fc_spectrum = nn.Sequential(
            nn.Linear(512*4 , 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 3 * points)
        )

    # block means SEPreActBlock or SEPreActBootleneck
    def _make_layer(self, block, planes, num_blocks, stride, reduction):
        strides = [stride] + [1] * (num_blocks - 1)  # like [1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, reduction))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _post_process_spectrm(self, spectrum):
        n = self.points
        amp = spectrum[:, :n]
        cos = spectrum[:, n:2*n]
        sin = spectrum[:, 2*n:]
        # amp = torch.sigmoid(amp)
        cos = 2 * torch.sigmoid(cos) - 1
        sin = 2 * torch.sigmoid(sin) - 1
        out = torch.stack([amp, cos, sin], dim=2)  # [batch_size, self.points, 3]
        return out

    def forward(self, x):
        out = self.convbnrelu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv_spectrum(out)
        out = out.view(out.size(0), -1)
        spectrum = self.fc_spectrum(out)
        spectrum = self._post_process_spectrm(spectrum)

        return spectrum


def SpectrumNet18(input_channel=1, points=183):
    return SpectrumNet(SEPreActBlock, [2, 2, 2, 2], input_channel=input_channel, points=points)


def SpectrumNet34(input_channel=1, points=183):
    return SpectrumNet(SEPreActBlock, [3, 4, 6, 3], input_channel=input_channel, points=points)


def SpectrumNet50(input_channel=1, points=183):
    return SpectrumNet(SEPreActBootleneck, [3, 4, 6, 3], input_channel=input_channel, points=points)


def SpectrumNet101(input_channel=1, points=183):
    return SpectrumNet(SEPreActBootleneck, [3, 4, 23, 3], input_channel=input_channel, points=points)


def SpectrumNet152(input_channel=1, points=183):
    return SpectrumNet(SEPreActBootleneck, [3, 8, 36, 3], input_channel=input_channel, points=points)


def demo():
    net = SpectrumNet50()
    spectrum = net((torch.randn(3, 1, 64, 64)))
    print(spectrum.size())

# demo()
