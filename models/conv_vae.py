from .vae import VAE, Flatten, Stack
import torch.nn as nn
from typing import Optional


class Conv_VAE(VAE):
    def __init__(self, channels: int, height: int, width: int, lr: int,
                 hidden_size: int, alpha: int,
                 dataset: Optional[str] = None,
                 save_images: Optional[bool] = None,
                 save_path: Optional[str] = None):
        super().__init__(hidden_size, alpha, lr,
                         dataset, save_images, save_path)
        # Our code now will look identical to the VAE class except that the
        # encoder and the decoder have been adjusted
        assert not height % 4 and not width % 4, "Choose height and width to "\
            "be divisible by 4"
        self.channels = channels
        self.height = height
        self.width = width
        final_height = (self.height//4-3)//2+1
        final_width = (self.width//4-3)//2+1
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 8, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 32x7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2),  # 64*3*3
            Flatten(),
            nn.Linear(128*final_height*final_width,
                      32*final_height*final_width),
            nn.ReLU(),
            nn.Linear(32*final_height*final_width, self.hidden_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 32*final_height *
                      final_width), nn.ReLU(),
            nn.BatchNorm1d(32*final_height * final_width),
            nn.Linear(32*final_height*final_width,
                      128*final_height*final_width),
            nn.ReLU(), nn.BatchNorm1d(128*final_height * final_width),
            Stack(128, 3, 3),
            nn.ConvTranspose2d(128, 64, 3, 2), nn.ReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 1, padding=1),
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 2, 2), nn.ReLU(), nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 2, 2), nn.BatchNorm2d(8),
            nn.Conv2d(8, self.channels, 3, padding=1), nn.Tanh()
        )
