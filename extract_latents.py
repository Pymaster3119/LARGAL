import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataparsing
import tqdm

# --- VAE model definitions
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, embd_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.in_proj = nn.Linear(embd_dim, 3 * embd_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embd_dim, embd_dim, bias=out_proj_bias)
        self.d_heads = embd_dim // n_heads
    def forward(self, x, casual_mask=False):
        batch_size, seq_len, d_embed = x.shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        # use reshape instead of view to handle possible non-contiguous tensors
        q = q.reshape(interim_shape)
        k = k.reshape(interim_shape)
        v = v.reshape(interim_shape)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        if casual_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape((batch_size, seq_len, d_embed))
        output = self.out_proj(output)
        return output

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    def forward(self, x):
        residual = x.clone()
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        # reshape to avoid errors when tensor is non-contiguous
        x = x.reshape((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.reshape((n, c, h, w))
        x += residual
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        residue = x.clone()
        x = self.groupnorm1(x)
        x = F.selu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = self.conv2(x)
        return x + self.residual_layer(residue)

class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    def forward(self, x):
        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        return x

class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mu = nn.Linear(512, 128)
        self.logvar = nn.Linear(512, 128)
        self.restarter = nn.Linear(128, 256)
    def forward(self, x):
        encoded = self.encoder(x)
        # encoded may be non-contiguous after convs; use reshape which handles that
        encoded = encoded.reshape(encoded.size(0), -1)
        mean = self.mu(encoded)
        log_variance = self.logvar(encoded)
        log_variance = torch.clamp(log_variance, -30, 20)
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        mean = mean + eps * std
        restarted = self.restarter(mean)
        # ensure correct shape even if restarted is non-contiguous
        restarted = restarted.reshape(restarted.size(0), 4, 8, 8)
        decoded = self.decoder(restarted)
        return decoded, mean, log_variance

# ----------------- extraction script -----------------

def load_model(path, device):
    """Load a model object or a state_dict from `path`.
    This handles checkpoints saved as full pickled Module objects (PyTorch <2.6 style)
    and the newer weights-only behavior introduced in PyTorch 2.6.

    Strategy:
    - Try to load the full object using safe_globals to allowlist `VAE` when available.
    - If that fails, try loading with weights_only=False (if available).
    - If loaded object is a dict, treat it as a state_dict and load into VAE().
    - Raise a descriptive error if all attempts fail.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Helper to attempt loading via torch.load with optional kwargs
    def try_torch_load(kwargs=None):
        kwargs = kwargs or {}
        try:
            return torch.load(path, map_location=device, **kwargs)
        except TypeError:
            # torch.load on older PyTorch may not accept some kwargs (e.g., weights_only)
            return torch.load(path, map_location=device)

    last_exc = None

    # 1) Try safe unpickling with VAE allowlisted (PyTorch >=2.6 provides safe_globals)
    try:
        try:
            with torch.serialization.safe_globals([VAE]):
                loaded = try_torch_load({'weights_only': False})
        except AttributeError:
            # safe_globals not available on older torch; just try loading with weights_only=False
            loaded = try_torch_load({'weights_only': False})

        if isinstance(loaded, nn.Module):
            model = loaded
            model.to(device)
            model.eval()
            return model

        if isinstance(loaded, dict):
            model = VAE()
            model.load_state_dict(loaded)
            model.to(device)
            model.eval()
            return model

    except Exception as e:
        last_exc = e

    # 2) Try loading as state_dict (weights-only) or with weights_only=False again
    try:
        state = try_torch_load({'weights_only': False})
        if isinstance(state, dict):
            model = VAE()
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            return model
        # If it's a Module instance, accept it
        if isinstance(state, nn.Module):
            model = state
            model.to(device)
            model.eval()
            return model
    except Exception as e2:
        last_exc = e2

    # Give helpful message including the last exception
    raise RuntimeError(f"Failed to load model from {path}. Last error: {last_exc}")


def extract_latents_for_split(split_name, data, model, device, out_dir, batch_size=64):
    dataset = dataparsing.AugmentedDataset(data, transforms=False, returnArea=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_latents = []
    all_labels = []
    all_areas = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].float().to(device)
            labels = batch[1].long().cpu().numpy()
            areas = batch[2].float().cpu().numpy()
            # if inputs are in 0..255, scale to 0..1
            if inputs.max() > 2.0:
                inputs = inputs / 255.0
            recon, mu_noisy, logvar = model(inputs)
            # get deterministic mean from encoder directly (bypass sampling)
            encoded = model.encoder(inputs)
            # use reshape to handle possible non-contiguous tensors
            encoded = encoded.reshape(encoded.size(0), -1)
            mu = model.mu(encoded)
            all_latents.append(mu.cpu().numpy())
            all_labels.append(labels)
            all_areas.append(areas)

    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    areas = np.concatenate(all_areas, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    lat_path = os.path.join(out_dir, f"latents_{split_name}.npy")
    lab_path = os.path.join(out_dir, f"labels_{split_name}.npy")
    area_path = os.path.join(out_dir, f"areas_{split_name}.npy")
    np.save(lat_path, latents)
    np.save(lab_path, labels)
    np.save(area_path, areas)
    print(f"Saved {split_name} latents -> {lat_path} (shape={latents.shape})")
    return lat_path, lab_path


device = torch.device("cuda")
model_path = "VAEModelCubicFit.pth"
out_dir = "latents"

print(f"Using device: {device}")
model = load_model(model_path, device)

splits = {
    'traincubicfit': dataparsing.train,
    'valcubicfit': dataparsing.val,
    'testcubicfit': dataparsing.test,
}

results = {}
for name, data in tqdm.tqdm(splits.items()):
    print(f"Processing split: {name} (N={len(data[0])})")
    lat_path, lab_path = extract_latents_for_split(name, data, model, device, out_dir)
    results[name] = (lat_path, lab_path)

print("Done. Saved latents for splits:")
for k, v in results.items():
    print(f"  {k}: {v[0]}, {v[1]}")
