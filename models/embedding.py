import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, points=183, laten_dim=128):
        super(Embedding, self).__init__()
        self.embedding_mu=nn.Sequential(
            nn.Linear(points*2,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, laten_dim)
        )
        self.embedding_var = nn.Sequential(
            nn.Linear(points * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, laten_dim)
        )

    def forward(self, spectrum):
        # input spectrum shape: [batch_size, 183, 2]
        spectrum = spectrum.view(spectrum.shape[0],-1)
        mu = self.embedding_mu(spectrum)
        log_var = self.embedding_var(spectrum)
        return mu, log_var


def demo():
    spectrum=torch.rand(3,183,2)
    embed = Embedding(points=183, laten_dim=128)
    mu, log_var = embed(spectrum)
    print(mu.shape)
    print(log_var.shape)


# demo()
