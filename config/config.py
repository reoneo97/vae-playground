from pydantic import BaseModel
from typing import Optional, Union
import yaml


class TrainConfig(BaseModel):
    max_epochs: int
    auto_lr_find: Union[bool, int]
    gpus: int


class VAEConfig(BaseModel):
    hidden_size: int
    alpha: int
    dataset: str
    save_images: Optional[bool] = False
    lr: Optional[float] = None
    save_path: Optional[str] = None


class ConvVAEConfig(VAEConfig):
    channels: int
    height: int
    width: int


class Config(BaseModel):
    model_config: Union[VAEConfig, ConvVAEConfig]
    train_config: TrainConfig
    model_type: str


def load_config(path="config.yaml"):
    config = yaml.load(open(path), yaml.SafeLoader)
    model_type = config['model_type']
    if model_type == "vae":
        model_config = VAEConfig(**config["model_params"])
    elif model_type == "conv-vae":
        model_config = ConvVAEConfig(**config["model_params"])

    train_config = TrainConfig(**config["training_params"])
    config = Config(model_config=model_config, train_config=train_config,
                    model_type=model_type)

    return config


config = load_config()
