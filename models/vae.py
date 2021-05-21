import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from torchvision.datasets import MNIST,FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
from torchvision.utils import save_image
import os
from torch.optim import Adam


class VAE(pl.LightningModule):
    def __init__(self,alpha = 1, dataset = "mnist"):
        #Autoencoder only requires 1 dimensional argument since input and output-size is the same
        
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784,196),nn.ReLU(),nn.BatchNorm1d(196,momentum = 0.7),
                                     nn.Linear(196,49),nn.ReLU(),nn.BatchNorm1d(49,momentum = 0.7),
                                     nn.Linear(49,28),nn.LeakyReLU())
        self.hidden2mu = nn.Linear(28,28)
        self.hidden2log_var = nn.Linear(28,28)
        self.alpha = alpha
        self.decoder = nn.Sequential(nn.Linear(28,49),nn.ReLU(),
                                     nn.Linear(49,196),nn.ReLU(),
                                     nn.Linear(196,784),nn.Tanh())
        self.data_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,),std=(0.5,))])
        self.dataset = dataset
    def encode(self,x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu,log_var

    def reparametrize(self,mu,log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size = (mu.size(0),mu.size(1)))
        z= z.type_as(mu)
        return mu + sigma*z

    def decode(self,x):
        x  = self.decoder(x)
        return x
    
    def training_step(self,batch,batch_idx):
        x,_ = batch
        batch_size = x.size(0)
        x = x.view(batch_size,-1)
        mu,log_var = self.encode(x)

        kl_loss =  (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = 1)).mean(dim =0)        
        hidden = self.reparametrize(mu,log_var)
        x_out = self.decode(hidden)
    
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x,x_out)
        #print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss


        self.log('train_loss',loss,on_step = False,on_epoch = True,prog_bar = True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,_ = batch
        batch_size = x.size(0)  
        x = x.view(batch_size,-1)
        mu,log_var = self.encode(x)

        kl_loss =  (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = 1)).mean(dim =0)        
        hidden = self.reparametrize(mu,log_var)
        x_out = self.decode(hidden)

        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x,x_out)
        #print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss
        self.log('val_kl_loss',kl_loss,on_step = False,on_epoch = True)
        self.log('val_recon_loss',recon_loss,on_step = False,on_epoch = True)
        self.log('val_loss',loss,on_step = False,on_epoch = True)
        return x_out,loss

    def validation_epoch_end(self,outputs):
        if not os.path.exists('vae_images'):
            os.makedirs('vae_images')
        choice = random.choice(outputs)
        output_sample = choice[0]
        output_sample = output_sample.reshape(-1, 1, 28, 28)
        output_sample = self.scale_image(output_sample)
        save_image(output_sample, f"vae_images/epoch_{self.current_epoch+1}.png")

    def configure_optimizers(self):
        return Adam(self.parameters(),lr = 1e-3)
        
    def forward(self,x):
        batch_size = x.size(0)
        x = x.view(batch_size,-1)
        mu,log_var = self.encode(x)
        hidden = self.reparametrize(mu,log_var)
        return self.decoder(hidden)

    # Functions for dataloading
    def train_dataloader(self):
        if self.dataset == "mnist":
            train_set = MNIST('data/',download = True,train = True,transform=self.data_transform)
        elif self.dataset == "fashion-mnist":
            train_set = FashionMNIST('data/',download = True,train = True,transform=self.data_transform) 
        return DataLoader(train_set,batch_size=64)

    def val_dataloader(self):
        if self.dataset == "mnist":
            val_set  = MNIST('data/',download = True,train = False,transform=self.data_transform)
        elif self.dataset == "fashion-mnist":
            val_set  = FashionMNIST('data/',download = True,train = False,transform=self.data_transform)    
        return DataLoader(val_set,batch_size=64)

    def scale_image(self,img):
        out = (img + 1) / 2
        return out

    def interpolate(self):
        pass