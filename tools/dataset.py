import os
import subprocess
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def imagenette_loaders(config):
    if not os.path.exists('./imagenette2-160'):
        subprocess.run(['wget', 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'], check=True)
        subprocess.run(['tar', '-xvzf', 'imagenette2-160.tgz'], check=True)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Normalizes between [0,1]
        transforms.Resize((160, 160))
    ])

    train_dataset = CustomDataset(image_dir='./imagenette2-160/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    test_dataset = CustomDataset(image_dir='./imagenette2-160/val', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    return train_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(f"{image_dir}/**/*.JPEG", recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image