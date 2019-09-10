'''
    writer: dororongju
    github: https://github.com/djkim1991/DCGAN/issues/1
'''
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.nz = 100   # 노이즈 벡터의 크기
        self.nc = 1     # 채널의 수
        self.nf = 64    # 필터 조정

        self.layer = nn.Sequential(
            # 입력값은 Z이며 Transposed Convolution을 거칩니다.
            nn.ConvTranspose2d(self.nz, self.nf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.nf * 10),
            nn.ReLU(True),

            # (nf * 10) x 4 x 4
            nn.ConvTranspose2d(self.nf * 8, self.nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU(True),

            # (nf * 4) x 8 x 8
            nn.ConvTranspose2d(self.nf * 4, self.nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.ReLU(True),

            # (nf * 2) x 16 x 16
            nn.ConvTranspose2d(self.nf * 2, self.nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(True),

            # nf x 32 x 32
            nn.ConvTranspose2d(self.nf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        """
        :param x:   input tensor    [batch_size * noise_size]
        :return:    output tensor   [batch_size * 1 * 28 * 28]
        """

        x = self.layer(x)

        return x.view(-1, 1, 28, 28)
