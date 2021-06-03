
from pytorch_lightning import Trainer
from models import VAE, Conv_VAE
import yaml


def load_config(model_type, path="config.yaml"):
    config = yaml.load(open(path), yaml.SafeLoader)
    if model_type == "vae":
        return config["training_params"], config["vae_model_params"]
    elif model_type == "conv-vae":
        return config["training_params"], config["conv_vae_model_params"]

if __name__ == "__main__":
    model_type = "conv-vae"
    training_params, model_params = load_config(model_type)

    model = Conv_VAE(**model_params)
    print("==== Model Architecture ====")
    print(model)
    trainer = Trainer(**training_params)
    trainer.fit(model)
