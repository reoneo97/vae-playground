from models import vae_models
import os
import torch
from torch.nn.functional import interpolate
from torchvision.transforms import Resize, ToPILImage, Compose
from torchvision.utils import make_grid

MODEL_DIR = "./saved_models/"


def load_model(filename):
    model_type = filename.split("_")[0]
    model = vae_models[model_type].load_from_checkpoint(MODEL_DIR + filename)
    model.eval()
    return model


def parse_model_file_name(file_name):
    # Hard Coded Parsing based on the filenames that I use
    substrings = file_name.split(".")[0].split("_")
    name, alpha, dim = substrings[0], substrings[2], substrings[4]
    new_name = ""
    if name == "vae":
        new_name += "Vanilla VAE"
    elif name == "conv-vae":
        new_name += "Convolutional VAE"
    new_name += f" | alpha={alpha}"
    new_name += f" | dim={dim}"
    return new_name


def canvas_to_tensor(canvas):
    """
    Convert Image of RGBA to single channel B/W and convert from numpy array
    to a PyTorch Tensor of [1,1,28,28]
    """
    img = canvas.image_data
    img = img[:, :, :-1]  # Ignore alpha channel
    img = img.mean(axis=2)
    img = img/255
    img = img*2 - 1.
    img = torch.FloatTensor(img)
    tens = img.unsqueeze(0).unsqueeze(0)
    tens = interpolate(tens, (28, 28))
    return tens


def tensor_to_img(tens):
    if tens.ndim == 4:
        tens = tens.squeeze(0)
    transform = Compose([
        ToPILImage()
    ])
    img = transform(tens)
    return img


def resize_img(img, w, h):
    return img.resize((w, h))


def perform_interpolation(model, tens1, tens2):
    output, dist1, dist2 = model.interpolate(tens1, tens2)
    output = (output+1)/2
    grid = make_grid(output, nrow=10)
    img = tensor_to_img(grid)
    return resize_img(img, 1500, 300)
