import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=28 * 8,
                               kernel_size=7, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=28 * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=28 * 8, out_channels=28 * 4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=28 * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=28 * 4, out_channels=1,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x.view(-1, 100, 1, 1)
        x = self.layer(x)

        return x
