
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from models import VAE


if __name__ == "__main__":
    model =  VAE(alpha = 50,dataset="fashion-mnist")
    print("==== Model Architecture ====")
    print(VAE)
    trainer = Trainer(gpus = 1,auto_lr_find=True,max_epochs=25)
    trainer.fit(VAE)
