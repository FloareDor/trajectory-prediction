"""Small PyTorch models used by the local training path."""

from __future__ import annotations

import torch
from torch import nn


class HistoryGRU(nn.Module):
    """A compact history-only predictor with a configurable future horizon."""

    def __init__(self, hidden_size: int = 64, future_steps: int = 20):
        super().__init__()
        self.future_steps = future_steps
        self.encoder = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, future_steps * 2),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(history)
        # Learn a correction to a physically sensible extrapolation. This
        # makes the first model stable on small local datasets while still
        # allowing it to learn turns and acceleration from the history.
        residual = self.decoder(hidden[-1]).view(history.shape[0], self.future_steps, 2)
        velocity = history[:, -1] - history[:, -2]
        steps = torch.arange(1, self.future_steps + 1, device=history.device, dtype=history.dtype).view(1, -1, 1)
        baseline = history[:, -1:, :] + steps * velocity[:, None, :]
        return baseline + residual


class SocialMapPredictor(nn.Module):
    """History model augmented with masked neighbor and map context."""

    def __init__(self, hidden_size: int = 64, future_steps: int = 20, modes: int = 1):
        super().__init__()
        self.future_steps = future_steps
        self.modes = modes
        self.history_encoder = nn.GRU(2, hidden_size, batch_first=True)
        self.neighbor_encoder = nn.Sequential(nn.Linear(2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.map_encoder = nn.Sequential(nn.Linear(2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, modes * future_steps * 2 + modes),
        )

    def forward(
        self,
        history: torch.Tensor,
        neighbors: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        map_polylines: torch.Tensor | None = None,
        map_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.history_encoder(history)
        target_context = hidden[-1]
        if neighbors is None:
            social = torch.zeros_like(target_context)
        else:
            encoded = self.neighbor_encoder(neighbors)
            if neighbor_mask is not None:
                encoded = encoded * neighbor_mask.unsqueeze(-1).float()
                denominator = neighbor_mask.float().sum(dim=(1, 2), keepdim=False).clamp_min(1.0).unsqueeze(-1)
                social = encoded.sum(dim=(1, 2)) / denominator
            else:
                social = encoded.mean(dim=(1, 2))
        if map_polylines is None:
            map_context = torch.zeros_like(target_context)
        else:
            encoded_map = self.map_encoder(map_polylines)
            if map_mask is not None:
                encoded_map = encoded_map * map_mask.unsqueeze(-1).float()
                denominator = map_mask.float().sum(dim=(1, 2), keepdim=False).clamp_min(1.0).unsqueeze(-1)
                map_context = encoded_map.sum(dim=(1, 2)) / denominator
            else:
                map_context = encoded_map.mean(dim=(1, 2))
        output = self.decoder(torch.cat([target_context, social, map_context], dim=-1))
        trajectory_size = self.modes * self.future_steps * 2
        residuals = output[:, :trajectory_size].view(-1, self.modes, self.future_steps, 2)
        velocity = history[:, -1] - history[:, -2]
        steps = torch.arange(1, self.future_steps + 1, device=history.device, dtype=history.dtype).view(1, 1, -1, 1)
        baseline = history[:, -1:, None, :] + steps * velocity[:, None, None, :]
        trajectories = baseline + residuals
        logits = output[:, trajectory_size:]
        return trajectories, logits
