"""Fixed-capacity models used by the Waymo component ablation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def constant_velocity_baseline(history: torch.Tensor, future_steps: int, dt: float = 0.1) -> torch.Tensor:
    """Extrapolate the current velocity from 8-D target state features."""
    position = history[:, -1, :2]
    velocity = history[:, -1, 2:4]
    steps = torch.arange(1, future_steps + 1, device=history.device, dtype=history.dtype)
    return position[:, None, :] + velocity[:, None, :] * (steps[None, :, None] * dt)


class ResidualHead(nn.Module):
    def __init__(self, input_size: int, future_steps: int, max_modes: int = 1):
        super().__init__()
        self.future_steps = future_steps
        self.max_modes = max_modes
        self.trajectory = nn.Linear(input_size, max_modes * future_steps * 2)
        self.mode_logits = nn.Linear(input_size, max_modes)

    def forward(self, context: torch.Tensor, history: torch.Tensor, num_modes: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        modes = self.max_modes if num_modes is None else num_modes
        if not 1 <= modes <= self.max_modes:
            raise ValueError(f"num_modes must be in [1, {self.max_modes}]")
        residual = self.trajectory(context).view(-1, self.max_modes, self.future_steps, 2)[:, :modes]
        baseline = constant_velocity_baseline(history, self.future_steps)
        return baseline[:, None] + residual, self.mode_logits(context)[:, :modes]


class TargetMLP(nn.Module):
    """A1: two-layer width-256 history-only MLP."""

    def __init__(self, history_steps: int = 11, future_steps: int = 80):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(history_steps * 8, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.head = ResidualHead(256, future_steps)

    def forward(self, batch: dict[str, torch.Tensor], **_: object) -> tuple[torch.Tensor, torch.Tensor]:
        history = batch["target_history"] * batch["history_valid"].unsqueeze(-1)
        return self.head(self.backbone(history.flatten(1)), history)


class TargetGRU(nn.Module):
    """A2: width-128 recurrent history baseline."""

    def __init__(self, future_steps: int = 80):
        super().__init__()
        self.gru = nn.GRU(8, 128, batch_first=True)
        self.head = ResidualHead(128, future_steps)

    def forward(self, batch: dict[str, torch.Tensor], **_: object) -> tuple[torch.Tensor, torch.Tensor]:
        history = batch["target_history"] * batch["history_valid"].unsqueeze(-1)
        output, _ = self.gru(history)
        # the current target state is always the last item
        return self.head(output[:, -1], history)


class VectorTransformer(nn.Module):
    """Shared vector encoder for A3--A7.

    Context switches only alter scene token masks.  All modules are present in
    every instance, which makes A3--A6 exact parameter-count controls.
    """

    def __init__(
        self,
        history_steps: int = 11,
        future_steps: int = 80,
        d_model: int = 128,
        num_heads: int = 4,
        temporal_layers: int = 2,
        scene_layers: int = 3,
        dropout: float = 0.1,
        max_modes: int = 6,
        use_neighbors: bool = True,
        use_map: bool = True,
        num_modes: int = 1,
    ):
        super().__init__()
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.use_neighbors = use_neighbors
        self.use_map = use_map
        self.num_modes = num_modes
        self.agent_projection = nn.Linear(8, d_model)
        self.temporal_position = nn.Parameter(torch.zeros(1, history_steps, d_model))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, temporal_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )
        self.map_point_encoder = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU(),
        )
        self.agent_type_embedding = nn.Embedding(5, d_model)
        self.map_type_embedding = nn.Embedding(32, d_model)
        self.token_kind_embedding = nn.Embedding(3, d_model)
        scene_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.scene_encoder = nn.TransformerEncoder(
            scene_layer, scene_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )
        self.head = ResidualHead(d_model, future_steps, max_modes=max_modes)
        nn.init.normal_(self.temporal_position, std=0.02)

    def _encode_agents(self, features: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # features [B,A,T,8], valid [B,A,T]
        batch, agents, steps, _ = features.shape
        flat_features = features.reshape(batch * agents, steps, 8)
        flat_valid = valid.reshape(batch * agents, steps)
        exists = flat_valid.any(dim=-1)
        safe_valid = flat_valid.clone()
        safe_valid[~exists, 0] = True  # avoid all-masked attention rows
        encoded = self.agent_projection(flat_features) + self.temporal_position[:, :steps]
        encoded = self.temporal_encoder(encoded, src_key_padding_mask=~safe_valid)
        weights = flat_valid.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = pooled * exists.unsqueeze(-1)
        return pooled.view(batch, agents, -1), exists.view(batch, agents)

    def _encode_map(self, features: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.map_point_encoder(features)
        exists = valid.any(dim=-1)
        encoded = encoded.masked_fill(~valid.unsqueeze(-1), torch.finfo(encoded.dtype).min)
        pooled = encoded.max(dim=2).values
        pooled = torch.where(exists.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        return pooled, exists

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        use_neighbors: bool | None = None,
        use_map: bool | None = None,
        num_modes: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor_switch = self.use_neighbors if use_neighbors is None else use_neighbors
        map_switch = self.use_map if use_map is None else use_map
        modes = self.num_modes if num_modes is None else num_modes
        target_features = batch["target_history"][:, None]
        target_valid = batch["history_valid"][:, None]
        all_features = torch.cat([target_features, batch["neighbors"]], dim=1)
        all_valid = torch.cat([target_valid, batch["neighbor_valid"]], dim=1)
        agent_tokens, agent_exists = self._encode_agents(all_features, all_valid)
        agent_types = torch.cat([batch["target_type"][:, None], batch["neighbor_types"]], dim=1)
        kind_agent = torch.ones_like(agent_types)
        kind_agent[:, 0] = 0
        agent_tokens = agent_tokens + self.agent_type_embedding(agent_types.clamp(0, 4)) + self.token_kind_embedding(kind_agent)

        map_tokens, map_exists = self._encode_map(batch["map_polylines"], batch["map_valid"])
        kind_map = torch.full_like(batch["map_types"], 2)
        map_tokens = map_tokens + self.map_type_embedding(batch["map_types"].clamp(0, 31)) + self.token_kind_embedding(kind_map)
        tokens = torch.cat([agent_tokens, map_tokens], dim=1)
        scene_valid = torch.cat([agent_exists, map_exists], dim=1)
        if not neighbor_switch:
            scene_valid[:, 1:9] = False
        if not map_switch:
            scene_valid[:, 9:] = False
        scene = self.scene_encoder(tokens, src_key_padding_mask=~scene_valid)
        return self.head(scene[:, 0], batch["target_history"], modes)


def build_ablation_model(experiment_id: str, history_steps: int = 11, future_steps: int = 80) -> nn.Module:
    if experiment_id == "A1":
        return TargetMLP(history_steps, future_steps)
    if experiment_id == "A2":
        return TargetGRU(future_steps)
    settings = {
        "A3": (False, False, 1), "A4": (True, False, 1),
        "A5": (False, True, 1), "A6": (True, True, 1),
        "A7": (True, True, 6),
    }
    if experiment_id not in settings:
        raise ValueError(f"unknown learned experiment {experiment_id}")
    neighbors, static_map, modes = settings[experiment_id]
    return VectorTransformer(
        history_steps=history_steps, future_steps=future_steps,
        use_neighbors=neighbors, use_map=static_map, num_modes=modes,
    )


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none").sum(dim=-1)
    weights = valid.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def multimodal_loss(
    trajectories: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    classification_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Winner-take-all masked regression plus mode classification."""
    regression_error = F.smooth_l1_loss(trajectories, target[:, None].expand_as(trajectories), reduction="none").sum(dim=-1)
    weights = valid[:, None].to(regression_error.dtype)
    per_mode = (regression_error * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    winner = per_mode.argmin(dim=1)
    chosen = trajectories[torch.arange(len(winner), device=winner.device), winner]
    regression = masked_smooth_l1(chosen, target, valid)
    classification = F.cross_entropy(logits, winner)
    return regression + classification_weight * classification, winner


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
