import torch.nn as nn
import torch



class Image_Block(nn.Module):
    def __init__(self, im_size):
        super(Image_Block, self).__init__()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.W = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        x = x - self.W
        return x


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(
            self, n_feats, kernel_size,
            bias=False, bn=True, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, bias=bias, padding=1))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # pdb.set_trace()
        res += x

        return res


class Dense_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CNN_res_64_64(nn.Module):
    def __init__(self):
        super(CNN_res_64_64, self).__init__()
        num_feat = 8
        kernel_size = 3
        n_resblocks = 10
        self.imgblock = Image_Block(im_size=64)
        self.conv1 = Conv_Block(1, num_feat, kernel_size=3, padding=1)
        body_1 = [ResBlock(num_feat, kernel_size) for _ in range(n_resblocks)]
        self.conv_res1 = nn.Sequential(*body_1)

        self.conv2 = Conv_Block(num_feat, num_feat * 2, kernel_size=3, padding=1)
        body_2 = [ResBlock(num_feat * 2, kernel_size) for _ in range(n_resblocks)]
        self.conv_res2 = nn.Sequential(*body_2)

        self.conv3 = Conv_Block(num_feat * 2, num_feat * 4, kernel_size=3, padding=1)
        body_3 = [ResBlock(num_feat * 4, kernel_size) for _ in range(n_resblocks)]
        self.conv_res3 = nn.Sequential(*body_3)

        self.conv4 = Conv_Block(num_feat * 4, num_feat * 8, kernel_size=3, padding=1)
        body_4 = [ResBlock(num_feat * 8, kernel_size) for _ in range(n_resblocks)]
        self.conv_res4 = nn.Sequential(*body_4)

        # 4*4*128

        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.avg_pool = nn.AvgPool2d(2, stride=2)

        self.fc1 = Dense_Block(num_feat * 8 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 183 + 366)
        self.drop = nn.Dropout(p=0.5)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # pdb.set_trace()
        x = self.imgblock(x)
        # for par in self.imgblock.parameters():
        #     print(par)

        x = self.conv1(x)
        x = self.conv_res1(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.conv_res2(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.conv_res3(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.conv_res4(x)
        x = self.max_pool(x)

        x = x.view(x.size(0), -1)
        # x = self.drop(x)

        x = self.fc1(x)
        # x = self.sig(x)
        # x = self.drop(x)
        x = self.fc2(x)
        amp = x[:, :183]
        cos = x[:, 183:366]
        sin = x[:, 366:]
        amp = self.sig(amp)
        cos = 2 * self.sig(cos) - 1
        sin = 2 * self.sig(sin) - 1
        x = torch.stack((amp, cos, sin), dim=2)

        return x
