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
            #self._build_model(config['training'])
        except:
            exit(f'Training config error.')




    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad



    def _build_model(self, config):
        #self.G_loss = self.G_loss(self)
        #self.D_loss = self.D_loss(self)
        self.criterionGAN = nn.MSELoss()

        self.criterionL1 = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=config['lr'])
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=config['lr'])
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.G = self.G.to(device)
        # self.D = self.D.to(device)
        # self.criterionGAN = self.criterionGAN.to(device)

    def G_wgan(self.G, self.D, x1,x2, y): # pylint: disable=unused-argument
        #latents = torch.normal([minibatch_size] + G.input_shapes[0][1:])
        #labels = training_set.get_random_labels_tf(minibatch_size)
        fake_images_out = self.G(x1,x2, y)
        fake_scores_out = self.D(fake_images_out, y)
        loss = -fake_scores_out
        return loss

    def D_wgan(self.G, self.D, x1,x2, y): # Weight for the epsilon term, \epsilon_{drift}.
        real_list = []
        fake_list = []

        #latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
        fake_images_out = self.G(x1,x2, y)
        real_scores_out = self.D(x1,x2, y)
        fake_scores_out = self.D(fake_images_out, y)
        real_list.append(real_scores_out)
        fake_list.append(fake_scores_out)

        loss = fake_scores_out - real_scores_out

        return loss



    def backward_D(self,y_G, y):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((y_G, y), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        ######这里应该是得到fake的loss，但是wgan那里好像并没有区分fake，所以我先用了一样的wgan。但肯定不对劲
        #self.loss_D_fake = self.criterionGAN(pred_fake)
        self.loss_D_fake = self.D_wgan(self.G, self.D, x1,x2, y)
        # Real
        real_AB = torch.cat((x1, x2, y), 1)
        pred_real = self.netD(real_AB)
        #self.loss_D_real = self.criterionGAN(pred_real)
        self.loss_D_real = self.D_wgan(self.G, self.D, x1,x2, y)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self,y_G, y):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((y_G, y), 1)
        pred_fake = self.netD(fake_AB)
        ######这里也不对劲，就是backward和loss的计算方式好像对不上
        #self.loss_G_GAN = G_wgan(pred_fake)
        self.loss_G_GAN = G_wgan(self.G, self.D,x1,x2, y)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(y_G, y)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()



    def start(self):
        train_it = self.data.iterator('train')
        #dev_it = self.data.iterator('dev')
        # test_it = self.data.iterator('test')
        #msg, x1, x2, y = next(train_it)
        #print(msg)

        for epoch in range(self.epoch):
            y_G_train, train_acc = self.train(self.G, self.D, train_it, self.optimizer_G, self.optimizer_D, self.criterionGAN)
            y_G_dev, dev_acc = self.evaluate(self.G, self.D, dev_it, self.criterionGAN)
            y_G_test, test_acc = self.evaluate(self.G, self.D, test_it, self.criterionGAN)

        return

    def train(self.G, self.D, train_it, self.optimizer_G, self.optimizer_D, self.criterionGAN):
        acc = []
        y_G_list = []

        for data in range(self.batch_size):

                msg, x1, x2, y = next(train_it)
                y_G = self.G(x1, x2)
                y_G_list.append(y_G)
                

                self.set_requires_grad(self.D, True)
                self.optimizer_D.zero_grad()
                #loss_D = self.criterionGAN(y_D, y)
                #self.loss_D.backward()
                self.backward_D(y_G, y)
                self.optimizer_D.step()

                self.set_requires_grad(self.D, False)
                self.optimizer_G.zero_grad()
                #loss_G = self.criterionGAN(y_G, y)
                #self.loss_G.backward()
                self.backward_G(y_G, y)
                self.optimizer_G.step()

                acc.append(nn.MSELoss(y_G, y))
        train_acc = np.mean(acc)

        return y_G_list, train_acc


    def evaluate(self.G, self.D, test_it, self.criterionGAN):
        acc = []
        y_G_list = []

        for msg, x1, x2, y in test_it:
            y_G = self.G(x1, x2)
            y_D = self.D(y_G, y)

            y_G_list.append(y_G)
            acc.append(nn.MSELoss(y_G, y))
        test_acc = np.mean(acc)

        return y_G_list, test_acc