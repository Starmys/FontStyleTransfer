import torch
import torch.nn as nn

from .data import Data
from .generator.base_generator import Generator
from .discriminator.base_discriminator import Discriminator


class GAN(object):

    def __init__(self, config):
        try:
            print('Loading data...')
            self.data = Data(config['data'])
            self.G = Generator(config['generator'])
            self.D = Discriminator(config['discriminator'])
            self._build_model(config['training'])
        except:
            exit(f'Training config error.')

    def _build_model(self, config):
        self.criterionGAN = nn.MSELoss()
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=config['lr'])
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=config['lr'])
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.G = self.G.to(device)
        # self.D = self.D.to(device)
        # self.criterionGAN = self.criterionGAN.to(device)

    def start(self):
        # train_it = self.data.iterator('train')
        # dev_it = self.data.iterator('dev')
        # test_it = self.data.iterator('test')
        # msg, x1, x2, y = next(train_it)
        # print(msg)
        return
