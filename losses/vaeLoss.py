import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAELoss(nn.Module):
    def __init__(self, M_N):
        super(VAELoss, self).__init__()
        self.M_N = M_N

    def forward(self, out):
        """
               Computes the VAE loss function.
               KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
               :param args:
               :param kwargs:
               :return:
               """
        recons = out["reconstruct"]
        input = out["input"]
        mu = out["mu"]
        log_var = out["log_var"]

        kld_weight = self.M_N  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = kld_weight * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}
