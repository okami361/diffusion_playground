import torch
from torch import nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, config):
        super(ConvEncoder, self).__init__()
        self.config = config

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv_mu = nn.Conv2d(256, config['latent_channels'], kernel_size=3, stride=1, padding=1)
        self.conv_logvar = nn.Conv2d(256, config['latent_channels'], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (3,160,160) -> (32,80,80)
        x = F.relu(self.bn2(self.conv2(x)))  # (32,80,80) -> (64,80,80)

        x = F.relu(self.bn3(self.conv3(x)))  # (64,80,80) -> (128,40,40)
        x = F.relu(self.bn4(self.conv4(x)))  # (128,40,40) -> (256,40,40)

        mu = self.conv_mu(x)                # (256,40,40) -> (latent_ch,40,40)
        logvar = self.conv_logvar(x)        # (256,40,40) -> (latent_ch,40,40)
        return mu, logvar

class ConvDecoder(nn.Module):
    def __init__(self, config):
        super(ConvDecoder, self).__init__()
        self.config = config

        self.deconv1 = nn.Conv2d(config['latent_channels'], 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.Conv2d(128, 64, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv_final = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = F.relu(self.bn1(self.deconv1(z)))  # (lat_ch,40,40) -> (256,40,40)
        x = F.relu(self.bn2(self.deconv2(x)))  # (256,40,40) -> (128,80,80)

        x = F.relu(self.bn3(self.deconv3(x)))  # (128,80,80) -> (64,80,80)
        x = F.relu(self.bn4(self.deconv4(x)))  # (64,80,80) -> (32,160,160)

        x = torch.sigmoid(self.conv_final(x))  # (32,160,160) -> (3,160,160)
        return x

class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.encoder = ConvEncoder(config)
        self.decoder = ConvDecoder(config)
        self.step_control_nodes = None

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    