import torch
import torch.nn as nn
import torchvision.models as models
import VAE.VAE

class GAN(nn.Module):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.model_vae = VAE(config)
        self.model_discriminator = Discriminator(pretrained=True)

    def forward(self, x):
        recon_x, mu, logvar = self.model_vae(x)
        y = self.model_discriminator(recon_x)
        
        return recon_x, mu, logvar, y

    def vae(self, x):
        recon_x, mu, logvar = self.model_vae(x)
        return recon_x, mu, logvar
    
    def discriminator(self, x):
        y = self.model_discriminator(x)
        return y

class Discriminator(nn.Module):
    def __init__(self, pretrained=True):
        super(Discriminator, self).__init__()
        
        # Load a ResNet50 backbone, pretrained on ImageNet
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # remove the original classification layer
        
        # Add your own classification head for real/fake
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features from ResNet (without its original final layer)
        features = self.resnet(x)
        
        # Forward through your custom classification layer
        out = self.classifier(features)
        return out