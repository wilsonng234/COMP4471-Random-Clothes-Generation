import torch
import torch.nn as nn
    

class Discriminator(nn.Module):     #PatchGAN
    def __init__(self, in_channels, ndf=64, n_layers=3,        #first layer is 64
                 norm_layer=nn.BatchNorm2d):       #n_layers = number of mid convolution run
        super().__init__()
        layers = self.get_layers(in_channels, ndf, n_layers, norm_layer)
        self.model = nn.Sequential(*layers)

    def get_layers(self, in_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        CNNkernel_size = 4
        pad = 1
        in_channels = in_channels*2
        sequence = []
        sequence += [nn.Conv2d(in_channels,
                    ndf,    
                    kernel_size=CNNkernel_size,
                    stride=2,
                    padding=pad,
                    padding_mode="reflect",
                    ),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=CNNkernel_size, stride=2, padding=pad,padding_mode='reflect'),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=CNNkernel_size, stride=1, padding=pad,padding_mode='reflect'),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=CNNkernel_size, stride=1, padding=pad)]

        return sequence

    def forward(self, x,y):
        z = torch.cat([x, y], dim=1)
        return self.model(z)


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()