'''
    writer: dororongju
    github: https://github.com/djkim1991/DCGAN/issues/1
'''
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


class MNIST:
    def __init__(self):
        self.train_loader, self.test_loader = self.load_data()

    def load_data(self):
        transform = transforms.Compose([
            transforms.Scale(28),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_set = torchvision.datasets.MNIST(
            root='./data/mnist',
            train=True,
            download=False,
            transform=transform
        )

        test_set = torchvision.datasets.MNIST(
            root='./data/mnist',
            train=False,
            download=False,
            transform=transform
        )
        train_loader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)

        return train_loader, test_loader
