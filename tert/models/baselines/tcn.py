"""TCN latent estimator (RMA-style) over a 50-step proprioceptive history."""

import torch
import torch.nn as nn

MIN_HISTORY = 40


class TCNEncoder(nn.Module):
    def __init__(self, obs_dim: int = 48, history_length: int = 50, latent_dim: int = 12):
        super().__init__()
        if history_length < MIN_HISTORY:
            raise ValueError(
                f"history_length={history_length} is too short for this TCN; "
                f"the strided stack needs at least {MIN_HISTORY} steps"
            )
        self.history_length = history_length

        self.conv = nn.Sequential(
            nn.Conv1d(obs_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, obs_dim, history_length)).shape[1]

        self.head = nn.Sequential(nn.Linear(flat_dim, 32), nn.ReLU(), nn.Linear(32, latent_dim))

    def forward(self, history):
        """history (B, history_length, obs_dim) -> latent (B, latent_dim)."""
        return self.head(self.conv(history.transpose(1, 2)))
