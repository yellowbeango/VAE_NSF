import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
import numpy as np
from typing import List, Callable, Union, Any, TypeVar, Tuple

# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

__all__ = ['SpectrumVAE']

class SpectrumVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 points: int = 183,
                 **kwargs) -> None:
        super(SpectrumVAE, self).__init__()

        self.latent_dim = latent_dim
        self.points = points

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.conv_spectrum = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, self.points*3, 1)
        )

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def _post_process_spectrm(self, spectrum):
        n = self.points
        amp = spectrum[:, :n]
        cos = spectrum[:, n:2*n]
        sin = spectrum[:, 2*n:]
        amp = torch.sigmoid(amp)
        cos = 2 * torch.sigmoid(cos) - 1
        sin = 2 * torch.sigmoid(sin) - 1
        out = torch.stack([amp, cos, sin], dim=2)  # [batch_size, self.points, 3]
        return out

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        spectrum = self.conv_spectrum(result).squeeze(dim=-1).squeeze(dim=-1)
        spectrum = self._post_process_spectrm(spectrum)  # [N x 3*points]
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var, spectrum]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        mu, log_var, spectrum = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return{
            "recons": recons,
            "input": input,
            "mu": mu,
            "log_var": log_var,
            "predict_spectrum": spectrum
        }

    def spectrum_loss(self, predict: Tensor, label: Tensor):
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
        loss_amp = F.mse_loss(Amp_true, Amp_predict)
        # cos/sin MSE loss for phase
        loss_phi = 0.5*(F.mse_loss(phi_x_true, phi_x_predict) + F.mse_loss(phi_y_true, phi_y_predict))
        return loss_amp, loss_phi

    def loss_function(self,
                      args,
                      targets: Tensor,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args["recons"]
        input = args["input"]
        mu = args["mu"]
        log_var = args["log_var"]
        predict_spectrum = args["predict_spectrum"]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        amp_weight = kwargs["amp_weight"]
        phi_weight = kwargs["phi_weight"]

        recons_loss = F.mse_loss(recons, input)
        kld_loss = kld_weight * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        predict_amp_loss, predict_phi_loss = self.spectrum_loss(predict_spectrum, targets)
        predict_amp_loss = amp_weight * predict_amp_loss
        predict_phi_loss = phi_weight * predict_phi_loss

        loss = recons_loss + kld_loss + predict_amp_loss + predict_phi_loss
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss,
            'KLD': kld_loss,
            'predict_amp_loss': predict_amp_loss,
            'predict_phi_loss': predict_phi_loss,
        }

    def sample(self, z, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        # z = torch.randn(num_samples,
        #                 self.latent_dim)
        #
        # z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)
