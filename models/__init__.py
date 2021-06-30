from .vae import VAE, Flatten, Stack  # noqa: F401
from .conv_vae import Conv_VAE  # noqa: F401

__all__ = [
    'VAE', 'Flatten', 'Stack'
    'Conv_VAE',
]
vae_models = {
    "conv-vae": Conv_VAE,
    "vae": VAE
}
