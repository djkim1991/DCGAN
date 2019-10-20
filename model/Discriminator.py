'''
    writer: dororongju
    github: https://github.com/djkim1991/DCGAN/issues/1
'''
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # self.nc = 1     # 채널의 수
        # self.nf = 64    # 필터 조정

        # self.layer = nn.Sequential(
        #     nn.Conv2d(self.nc, self.nf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Conv2d(self.nf, self.nf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.nf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Conv2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.nf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Conv2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.nf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Conv2d(self.nf * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=28 * 4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=28 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=28 * 4, out_channels=28 * 8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=28 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=28 * 8, out_channels=1,
                      kernel_size=7, stride=1, padding=0,
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x:   input tensor    [batch_size * 1 * 28 * 28]
        :return:    possibility of that the image is real data
        """
        # x = x.view(-1, 28*28)
        x = self.layer(x)

        return x.view(-1)
