import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.down_1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.down_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.down_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.up_0 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.up_1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.up_2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.up_3 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.up_4 = nn.Sequential(
            nn.Conv2d(11, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):                           # [b,   2,  32,  32]
        h_1 = self.down_0(x)                        # [b,  16,  16,  16]
        h_2 = self.down_1(h_1)                      # [b,  32,   8,   8]
        h_3 = self.down_2(h_2)                      # [b,  64,   4,   4]
        h_4 = self.down_3(h_3)                      # [b, 128,   2,   2]
        h_5 = self.up_0(h_4)                        # [b,  64,   4,   4]
        h_6 = self.up_1(torch.cat((h_5, h_3), 1))   # [b,  32,   8,   8]
        h_7 = self.up_2(torch.cat((h_6, h_2), 1))   # [b,  16,  16,  16]
        h_8 = self.up_3(torch.cat((h_7, h_1), 1))   # [b,   8,  32,  32]
        return self.up_4(torch.cat((h_8, x), 1))    # [b,   1,  32,  32]