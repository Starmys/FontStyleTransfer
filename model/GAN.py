import os
import importlib

import yaml
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
        self.log_path = os.path.join('logs', self.name)
        os.makedirs(self.log_path, exist_ok=True)
        with open(os.path.join(self.log_path, 'training_config.yaml'), 'w') as f:
            f.write(yaml.safe_dump(config))

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
        self.batch_size = config['batch_size']
        self.g_d_rate = config['g_d_rate']
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=config['lr'])
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=config['lr'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running on {str(self.device).upper()}')
        self.G, self.D = self._to_device([self.G, self.D])
        self.gan_loss = config['loss']['gan_loss']
        self.l1_loss_weight = config['loss']['l1_loss_weight']
        self.l2_loss_weight = config['loss']['l2_loss_weight']

    def _get_batch_data(self, iterator):
        x, y = [], []
        for _ in range(self.batch_size):
            msg, x_element, y_element = next(iterator)
            x.append(x_element)
            y.append(y_element)
        return torch.cat(x, 0), torch.cat(y, 0)

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

    def backward_D(self, x, y, y_hat):
        # Fake; stop backprop to the generator by detaching y_hat
        fake_D_input = torch.cat((x, y_hat), 1)
        pred_fake = self.D(fake_D_input.detach())
        # Real
        real_D_input = torch.cat((x, y), 1)
        pred_real = self.D(real_D_input)
        # combine loss and calculate gradients
        if self.gan_loss == 'lsgan':
            loss_D = (F.softplus(pred_fake) + F.softplus(-pred_real)) * 0.5
        else:
            loss_D = (pred_fake - pred_real) * 0.5
        loss_D = torch.mean(loss_D)
        loss_D.backward()
        return loss_D

    def backward_G(self, x, y, y_hat):
        fake_D_input = torch.cat((x, y_hat), 1)
        pred_fake = self.D(fake_D_input)
        if self.gan_loss == 'lsgan':
            loss_G_GAN = F.softplus(-pred_fake)
        elif self.gan_loss == 'wgan':
            loss_G_GAN = -pred_fake
        else:
            loss_G_GAN = 0
        loss_aux = F.l1_loss(y, y_hat) * self.l1_loss_weight + F.mse_loss(y, y_hat) * self.l2_loss_weight
        loss_G = loss_G_GAN + loss_aux
        loss_G = torch.mean(loss_G)
        loss_G.backward()
        return loss_G

    def start(self):
        self.G.apply(self.initialize_parameters)
        self.D.apply(self.initialize_parameters)
        train_it = self.data.iterator('train')
        for epoch in range(self.epoch_num):
            # self.epoch = epoch
            epoch_tag = str(epoch).zfill(len(str(self.epoch_num - 1)))
            print(f'========== Epoch {epoch_tag} ==========')
            self.train(train_it)
            mse = self.evaluate(self.data.iterator('dev'), f'epoch{epoch_tag}')
            print(f"Avg.MSE = {mse}")
            with open(os.path.join(self.log_path, 'MSELoss.txt', 'a')) as f:
                f.write(f'{mse}\n')
        print(f"Test Avg.MSE: {self.evaluate(self.data.iterator('test'), 'test')}")

    def train(self, iterator):
        print(f'Training...')
        for i in range(self.iteration_num):
            x, y = self._get_batch_data(iterator)
            x, y = self._to_device([x, y])
            y_hat = self.G(x)                          # compute fake images: y_hat = G(x)
            # update D
            if i % self.g_d_rate == 0:
                self.set_requires_grad(self.D, True)   # enable backprop for D
                self.optimizer_D.zero_grad()           # set D's gradients to zero
                loss_D = self.backward_D(x, y, y_hat)  # calculate gradients for D
                self.optimizer_D.step()                # update D's weights
            # update G
            self.set_requires_grad(self.D, False)      # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()               # set G's gradients to zero
            loss_G = self.backward_G(x, y, y_hat)      # calculate gradients for G
            self.optimizer_G.step()                    # udpate G's weights
            # Log
            if i % 100 == 0:
                print(f'Iteration {i}: Loss_D = {loss_D.item()}, Loss_G = {loss_G.item()}')

    def evaluate(self, iterator, tag):
        print(f'Evaluating...')
        num = 0
        loss = 0
        with torch.no_grad():
            for msg, x, y in iterator:
                x, y = self._to_device([x, y])
                y_hat = self.G(x)
                loss += F.mse_loss(y_hat, y).item()
                x, y, y_hat = self._cpu([x, y, y_hat])
                self.data.plot_results(self.name, tag, msg, [x[:, i:i+1, :, :] for i in range(x.shape[1])] + [y, y_hat])
                num += 1
        print(f'Evaluation output: {os.path.join("logs", self.name, tag)}')
        return loss / num
