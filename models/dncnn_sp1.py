
from torch import nn
from utils.spectral_normalization import SpectralNorm

class DnCNN(nn.Module):
    def __init__(self, network= 'Dncnn', net_mode=1, depth=12, n_channels=64, image_channels=1, kernel_size=3, is_traing=True):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        self.is_traing = is_traing
        layers.append(SpectralNorm(nn.Conv2d(
            in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False)))
        layers.append(nn.ReLU())

        for _ in range(depth-1):
            layers.append(SpectralNorm(nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)))
            layers.append(nn.ReLU())

        layers.append(SpectralNorm(nn.Conv2d(
            in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False)))

        self.dncnn_3 = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn_3(x)