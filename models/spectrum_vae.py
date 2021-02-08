import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
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

        self.spectrum = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 4, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, self.points * 2),
        )

        self.spectrum_embed = nn.Sequential(
            nn.Linear(self.points * 2, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, self.latent_dim)
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

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        spectrum = self.spectrum(result)

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

    def reparameterize(self, mu: Tensor, logvar: Tensor, target: Tensor = None) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        if target != None:
            eps = torch.randn_like(std)
        else:
            eps = target.reshape(-1, self.points * 2)
            eps = self.spectrum_embed(eps)
            eps = eps + torch.randn_like(eps)  # add some noise, providing diversity
            eps = (eps - eps.mean()) / (eps.std())  # to N(0,1)
            # eps = (eps+torch.randn_like(eps))/math.sqrt(2)
        return eps * std + mu

    def forward(self, input: Tensor, target: Tensor, **kwargs):
        mu, log_var, predict_spectrum = self.encode(input)
        z = self.reparameterize(mu, log_var, target)
        recons = self.decode(z)
        _, _, generate_spectrum = self.encode(recons)

        # [self.decode(z), input, mu, log_var, spectrum]
        return {
            "recons": recons,
            "input": input,
            "target": target,
            "mu": mu,
            "log_var": log_var,
            "predict_spectrum": predict_spectrum.reshape(-1, self.points, 2),  # spectrum from the input data
            "generate_spectrum": generate_spectrum.reshape(-1, self.points, 2)  # spectrum from the generated data

        }

    def loss_function(self,
                      args,
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
        generate_spectrum = args["generate_spectrum"]
        target = args["target"]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        spectrum_weight = kwargs["spectrum_weight"]
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        predict_spectrum_loss = (target - predict_spectrum).norm(dim=-1).mean()
        generate_spectrum_loss = (target - generate_spectrum).norm(dim=-1).mean()

        loss = recons_loss + kld_weight * kld_loss + spectrum_weight * (predict_spectrum_loss + generate_spectrum_loss)
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss,
            'KLD': -kld_loss,
            'predict_spectrum_loss': predict_spectrum_loss,
            'generate_spectrum_loss': generate_spectrum_loss
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

        return self.forward(x)["recons"]
