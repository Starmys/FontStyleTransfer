import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Data


class GAN(object):

    def __init__(self, config):
        try:
            print('Loading data...')
            self.name = config['name']
            self.data = Data(config['data'])
            generator = importlib.import_module(f'model.generator.{config["generator"]}')
            discriminator = importlib.import_module(f'model.discriminator.{config["discriminator"]}')
            self.G = generator.Generator()
            self.D = discriminator.Discriminator()
            self._build_model(config['training'])
        except:
            exit(f'Training config error.')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _to_device(self, arr):
        return [tensor.to(self.device) for tensor in arr]

    def _cpu(self, arr):
        return [tensor.cpu() for tensor in arr]

    def _build_model(self, config):
        torch.manual_seed(config['seed'])
        self.epoch_num = config['epoch']
        self.iteration_num = config['iteration']
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=config['lr'])
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=config['lr'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running on {self.device.upper()}')
        self.G, self.D = self._to_device([self.G, self.D])
        self.gan_loss = config['loss']['gan_loss']
        self.l1_loss_weight = config['loss']['l1_loss_weight']
        self.l2_loss_weight = config['loss']['l2_loss_weight']

    def initialize_parameters(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self, x_1, x_2, y, y_hat):
        # Fake; stop backprop to the generator by detaching y_hat
        fake_D_input = torch.cat((x_1, x_2, y_hat), 1)
        pred_fake = self.D(fake_D_input.detach())
        # Real
        real_D_input = torch.cat((x_1, x_2, y), 1)
        pred_real = self.D(real_D_input)
        # combine loss and calculate gradients
        if self.gan_loss == 'lsgan':
            loss_D = (F.softplus(pred_fake) + F.softplus(-pred_real)) * 0.5
        else:
            loss_D = (pred_fake - pred_real) * 0.5
        loss_D.backward()
        return loss_D

    def backward_G(self, x_1, x_2, y, y_hat):
        fake_D_input = torch.cat((x_1, x_2, y_hat), 1)
        pred_fake = self.D(fake_D_input)
        if self.gan_loss == 'lsgan':
            loss_G_GAN = F.softplus(-pred_fake)
        elif self.gan_loss == 'wgan':
            loss_G_GAN = -pred_fake
        else:
            loss_G_GAN = 0
        loss_G = loss_G_GAN + F.l1_loss(y, y_hat) * self.l1_loss_weight + F.mse_loss(y, y_hat) * self.l2_loss_weight
        loss_G.backward()
        return loss_G

    def start(self):
        self.G.apply(self.initialize_parameters)
        self.D.apply(self.initialize_parameters)
        train_it = self.data.iterator('train')
        for epoch in range(self.epoch_num):
            epoch_tag = str(epoch).zfill(len(str(self.epoch_num - 1)))
            print(f'========== Epoch {epoch_tag} ==========')
            self.train(train_it)
            self.evaluate(self.data.iterator('dev'), f'epoch{epoch_tag}')
        self.evaluate(self.data.iterator('test'), 'test')

    def train(self, iterator):
        print(f'Training...')
        for i in range(self.iteration_num):
            msg, x_1, x_2, y = next(iterator)
            # x_1, x_2 = y, y  # for auto-encoder
            x_1, x_2, y = self._to_device([x_1, x_2, y])
            y_hat = self.G(x_1, x_2)                      # compute fake images: y_hat = G(x_1, x_2)
            # update D
            self.set_requires_grad(self.D, True)          # enable backprop for D
            self.optimizer_D.zero_grad()                  # set D's gradients to zero
            loss_D = self.backward_D(x_1, x_2, y, y_hat)  # calculate gradients for D
            self.optimizer_D.step()                       # update D's weights
            # update G
            self.set_requires_grad(self.D, False)         # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()                  # set G's gradients to zero
            loss_G = self.backward_G(x_1, x_2, y, y_hat)  # calculate gradients for G
            self.optimizer_G.step()                       # udpate G's weights
            # Log
            if i % 100 == 0:
                print(f'Iteration {i}: Loss_D = {loss_D.item()}, Loss_G = {loss_G.item()}')

    def evaluate(self, iterator, tag):
        print(f'Evaluating...')
        for msg, x_1, x_2, y in iterator:
            # x_1, x_2 = y, y  # for auto-encoder
            x_1, x_2 = self._to_device([x_1, x_2])
            y_hat = self.G(x_1, x_2)          # compute fake images: y_hat = G(x_1, x_2)
            x_1, x_2, y, y_hat = self._cpu([x_1, x_2, y, y_hat])
            self.data.plot_results(self.name, tag, msg, [x_1, x_2, y, y_hat])
        print(f'Log: logs/{self.name}/{tag}')
