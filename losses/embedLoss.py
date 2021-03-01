import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EmbedLoss(nn.Module):
    def __init__(self):
        super(EmbedLoss, self).__init__()

    def forward(self, predict_mu, predict_var, vae_mu, vae_var):
        """
        Compute the embedding mean/var and VAE generated mean/var loss
        :param predict_mu:
        :param predict_var:
        :param vae_mu:
        :param vae_var:
        :return:
        """
        mu_loss = F.smooth_l1_loss(predict_mu, vae_mu)
        var_loss = F.smooth_l1_loss(predict_var, vae_var)
        loss = mu_loss + var_loss
        return {'loss': loss, 'mu_Loss': mu_loss, 'var_loss': var_loss}
