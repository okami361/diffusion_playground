import datetime
import glob
from PIL import Image

def generate_config(exp_number,name, output_path, dataset, description, image_resolution, latent_resolution, latent_channels, batch_size, learning_rate, num_epochs):
    log_object = {
        'exp_number': exp_number,
        'name': name,
        'output_path': output_path,
        'dataset' : dataset,
        'start_time': str(datetime.datetime.now()),
        'end_time': None,
        'description': description,
        'image_resolution': image_resolution,
        'latent_resolution': latent_resolution,
        'latent_channels': latent_channels,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'summary': '',
        'loss': 9999,
        'test_loss': 9999
    }
    return log_object
