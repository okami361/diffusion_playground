import torch
import torch.nn as nn
import torchvision.models as models
from VAE.VAE import VAE

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
    
    def discriminator_features(self, x):
        features = self.model_discriminator.extract_feature_maps(x)
        return features


class Discriminator(nn.Module):
    def __init__(self, pretrained=True):
        super(Discriminator, self).__init__()
        
        # Load a ResNet50 backbone, pretrained on ImageNet
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the original classification layer
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])  # Up to the last convolutional block
        num_features = self.resnet.fc.in_features
        
        # Add your own classification head for real/fake
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Perform global average pooling
            nn.Flatten(),                 # Flatten to a vector
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract feature maps
        feature_maps = self.feature_extractor(x)
        
        # Pass through classification head
        y = self.classifier(feature_maps)
        return y

    def extract_feature_maps(self, x):
        return self.feature_extractor(x)