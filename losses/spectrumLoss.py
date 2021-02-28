import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectrumLoss(nn.Module):
    def __init__(self, amp_weight=1.0, phi_weight=0.1):
        super(SpectrumLoss, self).__init__()
        self.amp_weight = amp_weight
        self.phi_weight = phi_weight

    def forward(self, predict, label):
        # copy:
        # https://github.com/xyhbobby/DL_for_optical_design//utils/utils.py # L224
        Amp_true = label[:, :, 0]
        Amp_predict = predict[:, :, 0]
        Phi_true = label[:, :, 1]
        phi_x_true = torch.cos(2 * np.pi * Phi_true - np.pi)
        phi_y_true = torch.sin(2 * np.pi * Phi_true - np.pi)
        phi_x_predict = predict[:, :, 1]
        phi_y_predict = predict[:, :, 2]
        # MSE loss for amplitude
        loss_amp = F.mse_loss(Amp_predict, Amp_true)
        loss_phi = 0.5*(F.mse_loss(phi_x_predict, phi_x_true) + F.mse_loss(phi_y_predict, phi_y_true))

        loss_amp = self.amp_weight*loss_amp
        loss_phi = self.phi_weight* loss_phi
        loss = loss_phi+ loss_phi

        return {"loss": loss,
                "loss_amp": loss_amp,
                "loss_phi": loss_phi
                }
