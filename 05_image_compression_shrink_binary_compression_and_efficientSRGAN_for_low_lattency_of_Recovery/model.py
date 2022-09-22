from math import log2
import torch
from torch import nn
from torchvision.models.efficientnet import MBConv, FusedMBConv, MBConvConfig, FusedMBConvConfig


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)  # in_c * 4, H, W --> in_c, H*2, W*2
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=[32,16,64,40,80,112,192,320], num_blocks=[1,2,2,3,3,4,1], ratio=4):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels[0], kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[MBConv(MBConvConfig(1.0, 3, 1, num_channels[i], num_channels[i+1], num_blocks[i]), 0.2, None) for i in range(len(num_blocks))])
        self.convblock = ConvBlock(num_channels[-1], num_channels[0], kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsamples = nn.Sequential(*[UpsampleBlock(num_channels[0], 2) for _ in range(int(log2(ratio)))])
        self.final = nn.Conv2d(num_channels[0], in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)

def test():
    import time
    import numpy as np
    from fvcore.nn.flop_count import flop_count

    low_resolution = 256//4  # 96x96 -> 24x24
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            x = torch.randn((1, 3, low_resolution, low_resolution))
            gen = Generator(ratio=4)
            gen_out = gen(x)
            t0 = time.time()
            gen_out = gen(x)
            print(time.time()-t0)
            disc = Discriminator()
            disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)

        model_parameters = filter(lambda p: p.requires_grad, gen.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)

        print(sum(flop_count(gen, x)[0].values()))

if __name__ == "__main__":
    test()