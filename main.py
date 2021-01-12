import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os

from models import *
import IPython

# fashion_train= FashionMNIST('data/',download = True,train = True,transform=transforms.ToTensor())


if __name__ == "__main__":
    
    vae =  VAE(alpha=100)
    print("==== Model Architecture ====")
    print(vae)
    trainer = Trainer(gpus = 1,auto_lr_find=True,max_epochs=10)
    trainer.fit(vae)

    path = "./lightning_logs"
    os.system('tensorboard --logdir=' + path)