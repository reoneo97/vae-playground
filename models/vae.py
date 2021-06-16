import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import Adam

import os
from typing import Optional


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class VAE(pl.LightningModule):
    def __init__(self, hidden_size: int, alpha: int, lr: float,
                 dataset: Optional[str] = None,
                 save_images: Optional[bool] = None,
                 save_path: Optional[str] = None):
        """Init function for the VAE

        Args:

        hidden_size (int): Latent Hidden Size
        alpha (int): Hyperparameter to control the importance of
        reconstruction loss vs KL-Divergence Loss
        lr (float): Learning Rate, will not be used if auto_lr_find is used.
        dataset (Optional[str]): Dataset to used
        save_images (Optional[bool]): Boolean to decide whether to save images
        save_path (Optional[str]): Path to save images
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.save_path = save_path
        self.save_images = save_images
        self.lr = lr
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(784, 196), nn.ReLU(),
            nn.BatchNorm1d(196, momentum=0.7),
            nn.Linear(196, 49), nn.ReLU(),
            nn.BatchNorm1d(49, momentum=0.7),
            nn.Linear(49, hidden_size), nn.LeakyReLU()
        )
        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2log_var = nn.Linear(hidden_size, hidden_size)
        self.alpha = alpha
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 49), nn.ReLU(),
            nn.Linear(49, 196), nn.ReLU(),
            nn.Linear(196, 784), Stack(1, 28, 28),
            nn.Tanh()
        )
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5,))])
        self.dataset = dataset

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def decode(self, x):
        x = self.decoder(x)
        return x

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)
        return mu + sigma*z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        mu, log_var, x_out = self.forward(x)

        kl_loss = (-0.5*(1+log_var - mu**2 -
                         torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss

        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        mu, log_var, x_out = self.forward(x)

        kl_loss = (-0.5*(1+log_var - mu**2 -
                         torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return x_out, loss

    def validation_epoch_end(self, outputs):
        if not self.save_images:
            return
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        choice = random.choice(outputs)
        output_sample = choice[0]
        output_sample = output_sample.reshape(-1, 1, 28, 28)
        print("Mean:", output_sample.mean())
        output_sample = self.scale_image(output_sample)
        save_image(output_sample,
                   f"{self.save_path}/epoch_{self.current_epoch+1}.png")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr or self.learning_rate))

    def forward(self, x):
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        output = self.decoder(hidden)
        return mu, log_var, output

    # Functions for dataloading
    def train_dataloader(self):
        if self.dataset == "mnist":
            train_set = MNIST('data/', download=True,
                              train=True, transform=self.data_transform)
        elif self.dataset == "fashion-mnist":
            train_set = FashionMNIST(
                'data/', download=True, train=True,
                transform=self.data_transform)
        return DataLoader(train_set, batch_size=64, shuffle=True)

    def val_dataloader(self):
        if self.dataset == "mnist":
            val_set = MNIST('data/', download=True, train=False,
                            transform=self.data_transform)
        elif self.dataset == "fashion-mnist":
            val_set = FashionMNIST(
                'data/', download=True, train=False,
                transform=self.data_transform)
        return DataLoader(val_set, batch_size=64)

    def scale_image(self, img):
        out = (img + 1) / 2
        return out

    def interpolate(self, x1, x2):
        assert x1.shape == x2.shape
        width, height = x1.size()[-2], x1.size()[-1]
        if self.training:
            raise Exception(
                "This function should not be called when model is still "
                "in training mode. Use model.eval() before calling the function")
        mu1, lv1 = self.encode(x1)
        mu2, lv2 = self.encode(x2)
        z1 = self.reparametrize(mu1, lv1)
        z2 = self.reparametrize(mu2, lv2)
        weights = torch.arange(0.1, 0.9, 0.1)
        intermediate = [self.decode(z1)]
        for wt in weights:
            inter = torch.lerp(z1, z2, wt)
            intermediate.append(self.decode(inter))
        intermediate.append(self.decode(z2))
        out = torch.stack(intermediate, dim=2)
        out = out.view(-1, width, height)
        return out
