from pytorch_lightning import Trainer
from models import VAE, Conv_VAE
from config import config


def make_model(config):
    model_type = config.model_type
    model_config = config.model_config

    if model_type == "vae":
        return VAE(**model_config.dict())
    elif model_type == "conv-vae":
        return Conv_VAE(**model_config.dict())
    else:
        raise NotImplementedError("Model Architecture not implemented")


if __name__ == "__main__":

    model = make_model(config)
    train_config = config.train_config
    trainer = Trainer(**train_config.dict())
    lr_finder = trainer.tuner.lr_find(model)
    new_lr = lr_finder.suggestion()
    model.lr = new_lr
    trainer.fit(model)
