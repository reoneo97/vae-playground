
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from vae import *
import IPython



if __name__ == "__main__":
    
    vae =  VAE(alpha = 50)
    print("==== Model Architecture ====")
    print(vae)
    trainer = Trainer(gpus = 1,auto_lr_find=True,max_epochs=25)
    trainer.fit(vae)
