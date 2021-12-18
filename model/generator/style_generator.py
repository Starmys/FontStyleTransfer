import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 32
        self.latent_dim = 128
        self.skeleton_layes = 2
        self.style_layers = 2
        self.num_layers = self.skeleton_layes + self.style_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        filters = [16 * (2 ** (i + 1)) for i in range(4)][::-1]
        filters = [min(x, 128) for x in filters]

        init_channels = filters[0]
        filters = [init_channels, *filters]
        filters[-1] = 1

        in_out_pairs = zip(filters[:-1], filters[1:])

        self.to_initial_block = nn.ConvTranspose2d(self.latent_dim, init_channels, 4, 1, 0, bias=False)

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):

            block = GeneratorBlock(
                self.latent_dim,
                in_chan,
                out_chan,
                upsample = ind != 0,
                upsample_rgb = ind != (self.num_layers - 1)
            )
            self.blocks.append(block)

        self.mapping_network = StyleVectorizer(self.latent_dim)

    def forward(self, x):

        batch_size = x.shape[0]
        image_size = x.shape[-1]

        # noise = torch.FloatTensor(batch_size, image_size, image_size, 1).uniform_(0., 1.)
        # noise = noise.to(self.device)
        noise = None

        # x_1 = x[:, 1:2, :, :]
        # style_1 = self.mapping_network(x_1)[:, None, :]
        # x_2 = x[:, 2:3, :, :]
        # style_2 = self.mapping_network(x_2)[:, None, :]
        # styles = torch.cat([
        #     style_1.expand(-1, self.skeleton_layes, -1),
        #     style_2.expand(-1, self.style_layers, -1)
        # ], dim=1)
        styles = self.mapping_network(x)[:, None, :].expand(-1, self.skeleton_layes + self.style_layers, -1)

        avg_style = styles.mean(dim=1)[:, :, None, None]
        x = self.to_initial_block(avg_style)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block in zip(styles, self.blocks):
            x, rgb = block(x, rgb, style, noise)

        return F.tanh(rgb)


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 1  # 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        # self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        # self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        # inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        # noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        # noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        # x = self.activation(x + noise1)
        x = self.activation(x)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        # x = self.activation(x + noise2)
        x = self.activation(x)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class ConvMappingNetwork(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        latent_dim = latent_dim // 4
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(1024, latent_dim),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512, latent_dim),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, latent_dim),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        y = []
        x = self.conv_1(x)
        y.append(self.fc_1(x))
        x = self.conv_2(x)
        y.append(self.fc_2(x))
        x = self.conv_3(x)
        y.append(self.fc_3(x))
        x = self.conv_4(x)
        y.append(self.fc_4(x))
        return torch.cat(y, dim=-1)

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            EqualLinear(1024, latent_dim, 0.1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(latent_dim, latent_dim, 0.1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(latent_dim, latent_dim, 0.1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(latent_dim, latent_dim, 0.1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)
