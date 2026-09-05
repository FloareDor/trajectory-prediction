"""Training and evaluation helpers for the local smoke path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .baselines import constant_velocity
from .data import TrajectorySample
from .metrics import ade, best_of_k_ade, best_of_k_fde, fde
from .models import HistoryGRU, SocialMapPredictor


class SampleDataset(Dataset):
    def __init__(self, samples: list[TrajectorySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "history": torch.from_numpy(sample.history),
            "future": torch.from_numpy(sample.future),
            "neighbors": torch.from_numpy(sample.neighbors),
            "neighbor_mask": torch.from_numpy(sample.neighbor_mask),
            "map_polylines": torch.from_numpy(sample.map_polylines),
            "map_mask": torch.from_numpy(sample.map_mask),
        }


def train_history_model(
    train_samples: list[TrajectorySample],
    validation_samples: list[TrajectorySample],
    output_dir: str | Path,
    epochs: int = 20,
    hidden_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> tuple[HistoryGRU, dict[str, object]]:
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    future_steps = train_samples[0].future.shape[0]
    model = HistoryGRU(hidden_size, future_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(SampleDataset(train_samples), batch_size=min(32, len(train_samples)), shuffle=True)
    loss_curve: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            prediction = model(batch["history"])
            loss = torch.nn.functional.smooth_l1_loss(prediction, batch["future"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        loss_curve.append(float(np.mean(epoch_losses)))
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in DataLoader(SampleDataset(validation_samples), batch_size=32):
            predictions.append(model(batch["history"]).numpy())
            targets.append(batch["future"].numpy())
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    metrics = {"model_ade": ade(pred, target), "model_fde": fde(pred, target)}
    baseline = np.stack([constant_velocity(sample.history, future_steps) for sample in validation_samples])
    metrics["constant_velocity_ade"] = ade(baseline, target)
    metrics["constant_velocity_fde"] = fde(baseline, target)
    metrics["train_loss_curve"] = loss_curve
    torch.save({"model_type": "history_gru", "model_state": model.state_dict(), "future_steps": future_steps, "hidden_size": hidden_size}, output_path / "best_model.pt")
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, metrics


def train_social_map_model(
    train_samples: list[TrajectorySample],
    validation_samples: list[TrajectorySample],
    output_dir: str | Path,
    epochs: int = 20,
    hidden_size: int = 64,
    modes: int = 3,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> tuple[SocialMapPredictor, dict[str, object]]:
    """Train the social/map model, including best-of-K multimodal loss."""
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    future_steps = train_samples[0].future.shape[0]
    model = SocialMapPredictor(hidden_size, future_steps, modes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(SampleDataset(train_samples), batch_size=min(16, len(train_samples)), shuffle=True)
    loss_curve: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            trajectories, logits = model(
                batch["history"], batch["neighbors"], batch["neighbor_mask"], batch["map_polylines"], batch["map_mask"]
            )
            per_mode = torch.linalg.vector_norm(trajectories - batch["future"][:, None], dim=-1).mean(dim=-1)
            winner = per_mode.argmin(dim=1)
            chosen = trajectories[torch.arange(len(winner)), winner]
            loss = torch.nn.functional.smooth_l1_loss(chosen, batch["future"])
            loss = loss + 0.1 * torch.nn.functional.cross_entropy(logits, winner)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        loss_curve.append(float(np.mean(epoch_losses)))
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in DataLoader(SampleDataset(validation_samples), batch_size=32):
            trajectories, _ = model(
                batch["history"], batch["neighbors"], batch["neighbor_mask"], batch["map_polylines"], batch["map_mask"]
            )
            predictions.append(trajectories.numpy())
            targets.append(batch["future"].numpy())
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    metrics = {
        "best_of_k_ade": float(np.mean([best_of_k_ade(p, t) for p, t in zip(pred, target)])),
        "best_of_k_fde": float(np.mean([best_of_k_fde(p, t) for p, t in zip(pred, target)])),
        "train_loss_curve": loss_curve,
    }
    torch.save(
        {"model_type": "social_map", "model_state": model.state_dict(), "future_steps": future_steps, "hidden_size": hidden_size, "modes": modes},
        output_path / "social_map_model.pt",
    )
    (output_path / "social_map_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, metrics
