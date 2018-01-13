import torch
import torch.nn as nn

import code


class VAE(nn.Module):
    def __init__(self, size):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.mean = nn.Linear(256, 64)

        self.log_var = nn.Linear(256, 64)

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, size),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode.
        out = self.encoder(x)

        # Mean and standard deviation.
        mean = self.mean(out)
        log_var = self.log_var(out)

        # Epsilon trick.
        out = code.reparameterize(self, mean, log_var)

        # Decode.
        out = self.decoder(out)

        return out

    def forward_mean(self, x):
        # Encode.
        out = self.encoder(x)

        # Mean.
        mean = self.mean(out)

        return mean

    def forward_log_var(self, x):
        # Encode.
        out = self.encoder(x)

        # Log variance.
        log_var = self.log_var(out)

        return log_var
