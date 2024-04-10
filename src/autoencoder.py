import torch
import torch.nn as nn
import torch

from .denoising_diffusion_process.backbones.simple_unet import m_Linear


class ComplexAutoencoder(nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 n_layers, 
                 dropout_pra):
        super().__init__()
 
        self.encoder = nn.Sequential(
            m_Linear(input_dim, hidden_dim//2),  # input size same as input data
            nn.Dropout(p=dropout_pra),
            nn.GELU(),
            m_Linear(hidden_dim//2, hidden_dim),
            nn.Dropout(p=dropout_pra),
            nn.GELU(),
            *[m_Linear(hidden_dim, hidden_dim), nn.GELU()] * n_layers,
            m_Linear(hidden_dim, hidden_dim),
        )

        self.decoder = nn.Sequential(
            *[m_Linear(hidden_dim, hidden_dim), nn.GELU()] * n_layers,
            m_Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            m_Linear(hidden_dim//2, input_dim),
        )


    def forward(self, x):
        encoded = self.encoder(x)  # @ * 2 * 64

        reconstructed = self.decoder(encoded.view(encoded.size(0), 2, -1))

        return reconstructed
    

    def encode(self, x):
        encoded = self.encoder(x)
        return self.compute_abs(encoded)
    

    def encode_complex(self, x):
        return self.encoder(x)
    

    def decode_complex(self, z):
        return self.decoder(z)


    @staticmethod
    def compute_abs(x):
        real_part = x[:, 0, :] ** 2
        imag_part = x[:, 1, :] ** 2
        abs_value = torch.sqrt(real_part + imag_part)
        return abs_value


class LocationClassifier(nn.Module):

    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 n_layers,
                 dropout_pra, 
                 n_locs):
        super().__init__()

        self.n_locs = n_locs

        self.encoder = nn.Sequential(
            m_Linear(input_dim, hidden_dim//2),  # input size same as input data
            nn.Dropout(p=dropout_pra),
            nn.GELU(),
            m_Linear(hidden_dim//2, hidden_dim),
            nn.Dropout(p=dropout_pra),
            nn.GELU(),
            *[m_Linear(hidden_dim, hidden_dim), nn.GELU()] * n_layers,
            m_Linear(hidden_dim, hidden_dim),
        )

        self.fc_out = nn.Linear(hidden_dim, self.n_locs)


    def forward(self, x):
        encoded = self.encoder(x)                             

        encoded_abs = self.compute_abs(encoded)

        output = self.fc_out(encoded_abs)

        return output
    

    @staticmethod
    def compute_abs(x):
        real_part = x[:, 0, :] ** 2
        imag_part = x[:, 1, :] ** 2
        abs_value = torch.sqrt(real_part + imag_part)
        return abs_value

