from pytorch_lightning import Trainer
from models import vae_models
from config import config
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os

def make_model(config):
    model_type = config.model_type
    model_config = config.model_config

    if model_type not in vae_models.keys():
        raise NotImplementedError("Model Architecture not implemented")
    else:
        return vae_models[model_type](**model_config.dict())


if __name__ == "__main__":
    model = make_model(config)
    train_config = config.train_config
    logger = TensorBoardLogger(**config.log_config.dict())
    trainer = Trainer(**train_config.dict(), logger=logger,
                      callbacks=LearningRateMonitor())
    if train_config.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model)
        new_lr = lr_finder.suggestion()
        print("Learning Rate Chosen:", new_lr)
        model.lr = new_lr
        trainer.fit(model)
    else:
        trainer.fit(model)
    if not os.isdir("./saved_models"):
        os.mkdir("./saved_models")
    trainer.save_checkpoint(
        f"saved_models/{config.model_type}_alpha_{config.model_config.alpha}_dim_{config.model_config.hidden_size}.ckpt")
