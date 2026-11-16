import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*layers)


class DWConvBNReLU(ConvBNReLU):
    def __init__(self, channels, kernel_size=5, stride=1):
        nn.Sequential.__init__(
            self,
            nn.Conv2d(channels, channels, kernel_size, stride, kernel_size // 2, groups=channels, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


class PersonDetector(nn.Module):
    stride = 4

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(1, 16, 3, 2),
            ConvBNReLU(16, 32, 1),
            DWConvBNReLU(32, 5, 2),
            ConvBNReLU(32, 64, 1),
            DWConvBNReLU(64),
            ConvBNReLU(64, 80, 1),
            DWConvBNReLU(80),
            ConvBNReLU(80, 96, 1),
            DWConvBNReLU(96),
            ConvBNReLU(96, 96, 1),
            DWConvBNReLU(96),
            ConvBNReLU(96, 112, 1),
            DWConvBNReLU(112),
            ConvBNReLU(112, 112, 1),
            DWConvBNReLU(112),
            ConvBNReLU(112, 128, 1),
            DWConvBNReLU(128),
            ConvBNReLU(128, 128, 1),
            DWConvBNReLU(128),
            ConvBNReLU(128, 128, 1),
            ConvBNReLU(128, 64, 1),
            DWConvBNReLU(64, 3),
            nn.Conv2d(64, 5, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.net[-1].bias[4] = torch.log(torch.tensor(0.01 / 0.99)).item()

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = PersonDetector()
    out = model(torch.randn(1, 1, 120, 160))
    print("Modèle OK, sortie :", out.shape)  # Doit afficher torch.Size([1, 5, 30, 40])