import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
                 batch_size: int,
                 dataset: Optional[str] = None,
                 save_images: Optional[bool] = None,
                 save_path: Optional[str] = None, **kwargs):
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
        self.save_path = f'{save_path}/{kwargs["model_type"]}_images/'
        self.save_hyperparameters()
        self.save_images = save_images
        self.lr = lr
        self.batch_size = batch_size
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(784, 392), nn.BatchNorm1d(392), nn.LeakyReLU(0.1),
            nn.Linear(392, 196), nn.BatchNorm1d(196), nn.LeakyReLU(0.1),
            nn.Linear(196, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, hidden_size)
        )
        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2log_var = nn.Linear(hidden_size, hidden_size)
        self.alpha = alpha
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 196), nn.BatchNorm1d(196), nn.LeakyReLU(0.1),
            nn.Linear(196, 392), nn.BatchNorm1d(392), nn.LeakyReLU(0.1),
            nn.Linear(392, 784),
            Stack(1, 28, 28),
            nn.Tanh()
        )
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x:2*x-1.)])
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
        z = torch.randn_like(sigma)
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
        # print(x.mean(),x_out.mean())
        return x_out, loss

    def validation_epoch_end(self, outputs):
        if not self.save_images:
            return
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        choice = random.choice(outputs)
        output_sample = choice[0]
        output_sample = output_sample.reshape(-1, 1, 28, 28)
        # output_sample = self.scale_image(output_sample)
        save_image(
            output_sample,
            f"{self.save_path}/epoch_{self.current_epoch+1}.png",
            # value_range=(-1, 1)
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        lr_scheduler = ReduceLROnPlateau(optimizer,)
        return {
            "optimizer": optimizer, "lr_scheduler": lr_scheduler,
            "monitor": "val_loss"
        }

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
        return DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

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

        assert x1.shape == x2.shape, "Inputs must be of the same shape"
        if x1.dim() == 3:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        if self.training:
            raise Exception(
                "This function should not be called when model is still "
                "in training mode. Use model.eval() before calling the "
                "function")
        mu1, lv1 = self.encode(x1)
        mu2, lv2 = self.encode(x2)
        z1 = self.reparametrize(mu1, lv1)
        z2 = self.reparametrize(mu2, lv2)
        weights = torch.arange(0.1, 0.9, 0.1)
        intermediate = [self.decode(z1)]
        for wt in weights:
            inter = (1.-wt)*z1 + wt*z2
            intermediate.append(self.decode(inter))
        intermediate.append(self.decode(z2))
        out = torch.stack(intermediate, dim=0)
        return out, (mu1, lv1), (mu2, lv2)
