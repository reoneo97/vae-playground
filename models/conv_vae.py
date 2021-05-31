from .vae import VAE
import torch.nn as nn


class Conv_VAE(VAE):
    def __init__(self, channels: int, hidden_size: int, alpha: int = 1,
                 dataset: str = "mnist", save_images: bool = True):                
        super().__init__(hidden_size, alpha, dataset, save_images)
        # Our code now will look identical to the VAE class except that the
        # encoder and the decoder have been adjusted
        self.channels = channels

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels,16,3),nn.ReLU(),
            nn

