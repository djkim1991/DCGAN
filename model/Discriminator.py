import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

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
        # x = x.view(-1, 28*28)
        x = self.layer(x)

        return x.view(-1)
