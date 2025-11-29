import numpy as np
import torch
import torch.nn.functional as F

#region Class-Balanced for ViT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_CLS = [4155 * 0.13, 4155 * 0.48, 4155 * 0.14, 4155 * 0.25]
NO_OF_CLASSES = 4
beta = (4155.0 - 1) / 4155.0
effective_num = 1.0 - np.power(beta, SAMPLES_PER_CLS)
precomputed_weights = (1.0 - beta) / np.array(effective_num)
precomputed_weights = precomputed_weights / np.sum(precomputed_weights) * NO_OF_CLASSES
precomputed_weights = torch.tensor(precomputed_weights).float().to(device)

def CB_loss(labels, logits, loss_type, weights):
    labels_one_hot = F.one_hot(labels, weights.size(0)).to(device).float()
    if loss_type == "normal":
        cb_loss = F.cross_entropy(input=logits, target=labels, weight=weights)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss

#endregion

#region VAE Loss Function
def VAE_loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
#endregion