from models import vae_models
import os

def load_model(filename):
    model_name = filename.split("_")[0]
    return model_name

def model_types():
    files = os.listdir("saved_models/")
    return files