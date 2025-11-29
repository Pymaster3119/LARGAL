import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = torch.zeros_like(mu)
        return mu, logvar

    def reparameterize(self, mu, logvar, train=True):
        if train:
            return mu
        cluster_centers_tensor = torch.stack([torch.ones_like(mu[0])])
        distances = torch.cdist(mu, cluster_centers_tensor)
        min_distances, _ = torch.min(distances, dim=1)
        epsilon = 1e-6
        std = 20 / (min_distances + epsilon)
        std = std.unsqueeze(1).expand_as(mu)
        eps = torch.randn_like(mu) * 20
        z = mu + std * eps
        return z

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 64, 16, 16)
        x = self.decoder(x)
        return x

    def forward(self, x, train=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, train)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

if __name__ == '__main__':
    device = torch.device('cpu')
    vae = VAE(latent_dim=512).to(device)
    inp = torch.randn(4, 3, 64, 64).to(device)
    with torch.no_grad():
        recon, mu, logvar = vae(inp, train=False)
    print('recon shape:', recon.shape)
    print('mu shape:', mu.shape)
    print('OK')
