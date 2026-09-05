"""Training, prediction, aggregation, and reporting for the motion ablation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .ablation_data import ShardedWaymoDataset, deterministic_partitions
from .ablation_metrics import paired_bootstrap_macro_ci, summarize_predictions
from .ablation_models import (
    build_ablation_model,
    constant_velocity_baseline,
    masked_smooth_l1,
    multimodal_loss,
    parameter_count,
)


EXPERIMENTS = {
    "A0": {"architecture": "constant_velocity", "neighbors": False, "map": False, "modes": 1},
    "A1": {"architecture": "mlp", "neighbors": False, "map": False, "modes": 1},
    "A2": {"architecture": "gru", "neighbors": False, "map": False, "modes": 1},
    "A3": {"architecture": "transformer", "neighbors": False, "map": False, "modes": 1},
    "A4": {"architecture": "transformer", "neighbors": True, "map": False, "modes": 1},
    "A5": {"architecture": "transformer", "neighbors": False, "map": True, "modes": 1},
    "A6": {"architecture": "transformer", "neighbors": True, "map": True, "modes": 1},
    "A7": {"architecture": "transformer", "neighbors": True, "map": True, "modes": 6},
}
COMPARISONS = {
    "architecture_gru_minus_mlp": ("A1", "A2"),
    "architecture_transformer_minus_gru": ("A2", "A3"),
    "neighbors_without_map": ("A3", "A4"),
    "neighbors_with_map": ("A5", "A6"),
    "map_without_neighbors": ("A3", "A5"),
    "map_with_neighbors": ("A4", "A6"),
    "combined_context": ("A3", "A6"),
    "multimodality": ("A6", "A7"),
}


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_dir": "experiments/waymo_ablation",
    "cache": {"train_manifest": "data/processed/waymo/train-manifest.json", "validation_manifest": "data/processed/waymo/validation-manifest.json"},
    "split_salt": "waymo-motion-ablation-v1",
    "screen": {"train_targets": 10_000, "dev_targets": 2_000, "seeds": [17]},
    "final": {"train_targets": 50_000, "dev_targets": 10_000, "seeds": [17, 29, 43]},
    "training": {
        "batch_size": 32, "gradient_accumulation": 2, "learning_rate": 3e-4,
        "weight_decay": 1e-4, "epochs": 40, "warmup_fraction": 0.05,
        "early_stopping_patience": 6, "gradient_clip": 1.0,
        "mixed_precision": True, "num_workers": 0,
    },
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_ablation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    user = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = _merge(DEFAULT_CONFIG, user)
    config["_config_path"] = str(config_path.resolve())
    config["_base_dir"] = str(config_path.resolve().parent)
    return config


def _resolve(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # keep relative paths anchored to the working directory
    return Path.cwd() / path


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def environment_metadata(device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "PyYAML", "pyarrow", "tensorflow", "waymo-open-dataset-tf-2-12-0"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {"name": properties.name, "total_memory_bytes": properties.total_memory, "cuda": torch.version.cuda}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "gpu": gpu,
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _slices(batch_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    angle = batch_arrays["heading_change"]
    return {
        "turning": np.isfinite(angle) & (angle >= np.deg2rad(15.0)),
        "straight": np.isfinite(angle) & (angle <= np.deg2rad(5.0)),
        "interactive": batch_arrays["interactive"].astype(bool),
        "intersection": batch_arrays["intersection"].astype(bool),
    }


def predict_dataset(model: torch.nn.Module | None, dataset: ShardedWaymoDataset, device: torch.device, batch_size: int, experiment_id: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    collected: dict[str, list[np.ndarray]] = {key: [] for key in (
        "trajectories", "logits", "future", "future_valid", "target_type", "cache_index",
        "origin", "heading", "ground_truth", "ground_truth_valid", "object_id",
        "heading_change", "interactive", "intersection",
    )}
    scenario_ids: list[str] = []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            if experiment_id == "A0":
                trajectory = constant_velocity_baseline(batch["target_history"], batch["future"].shape[1])[:, None]
                logits = torch.zeros((len(trajectory), 1), device=device)
            else:
                assert model is not None
                trajectory, logits = model(batch)
            collected["trajectories"].append(trajectory.float().cpu().numpy())
            collected["logits"].append(logits.float().cpu().numpy())
            for key in collected:
                if key in {"trajectories", "logits"}:
                    continue
                value = raw_batch[key]
                collected[key].append(value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
            scenario_ids.extend(str(value) for value in raw_batch["scenario_id"])
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    arrays["scenario_id"] = np.asarray(scenario_ids)
    summary, per_sample = summarize_predictions(
        arrays["trajectories"], arrays["logits"], arrays["future"], arrays["future_valid"],
        arrays["target_type"], _slices(arrays),
    )
    for key, value in per_sample.items():
        arrays[f"metric_{key}"] = value
    return summary, arrays


def _save_predictions(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float):
    warmup = max(1, round(total_steps * warmup_fraction))
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one(
    experiment_id: str,
    seed: int,
    train_dataset: ShardedWaymoDataset,
    dev_dataset: ShardedWaymoDataset,
    run_dir: Path,
    config: dict[str, Any],
) -> dict[str, object]:
    seed_everything(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = train_dataset.manifest["spec"]
    model = build_ablation_model(experiment_id, int(spec["history_steps"]), int(spec["future_steps"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    accumulation = int(training["gradient_accumulation"])
    loader = DataLoader(
        train_dataset, batch_size=int(training["batch_size"]), shuffle=True,
        num_workers=int(training["num_workers"]), pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation)
    scheduler = _make_scheduler(optimizer, optimizer_steps_per_epoch * int(training["epochs"]), float(training["warmup_fraction"]))
    amp_enabled = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    curves: list[dict[str, float]] = []
    best_metric = float("inf")
    stale = 0
    checkpoint_path = run_dir / "checkpoint.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for batch_index, raw_batch in enumerate(loader):
            batch = _move_batch(raw_batch, device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                trajectories, logits = model(batch)
                if trajectories.shape[1] == 1:
                    loss = masked_smooth_l1(trajectories[:, 0], batch["future"], batch["future_valid"])
                else:
                    loss, _ = multimodal_loss(trajectories, logits, batch["future"], batch["future_valid"])
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            losses.append(float(loss.detach()))
        dev_metrics, _ = predict_dataset(model, dev_dataset, device, int(training["batch_size"]), experiment_id)
        selected = float(dev_metrics["primary_macro_minade"])
        print(f"{experiment_id} seed {seed}: epoch {epoch}, dev minADE {selected:.4f}", flush=True)
        curves.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_macro_minade": selected, "learning_rate": scheduler.get_last_lr()[0]})
        (run_dir / "curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
        if selected < best_metric:
            best_metric = selected
            stale = 0
            torch.save({
                "experiment_id": experiment_id, "seed": seed, "model_state": model.state_dict(),
                "history_steps": int(spec["history_steps"]), "future_steps": int(spec["future_steps"]),
                "best_epoch": epoch, "best_dev_macro_minade": best_metric,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]):
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics, arrays = predict_dataset(model, dev_dataset, device, int(training["batch_size"]), experiment_id)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    metrics.update({
        "experiment_id": experiment_id, "seed": seed, "best_epoch": checkpoint["best_epoch"],
        "parameter_count": parameter_count(model), "peak_cuda_bytes": peak,
    })
    if peak > 7_500_000_000:
        raise RuntimeError(f"peak CUDA allocation {peak / 1e9:.2f} GB exceeds the 7.5 GB acceptance limit")
    _save_predictions(run_dir / "predictions.npz", arrays)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    frozen = {key: value for key, value in config.items() if not key.startswith("_")}
    frozen["run"] = {"experiment_id": experiment_id, "seed": seed, "train_manifest": str(train_dataset.manifest_path.resolve())}
    (run_dir / "frozen_config.yaml").write_text(yaml.safe_dump(frozen, sort_keys=True), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(environment_metadata(device), indent=2), encoding="utf-8")
    return metrics


def _baseline_run(dev_dataset: ShardedWaymoDataset, run_dir: Path, config: dict[str, Any]) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics, arrays = predict_dataset(None, dev_dataset, device, int(config["training"]["batch_size"]), "A0")
    metrics.update({"experiment_id": "A0", "seed": None, "parameter_count": 0, "peak_cuda_bytes": 0})
    _save_predictions(run_dir / "predictions.npz", arrays)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    frozen = {key: value for key, value in config.items() if not key.startswith("_")}
    frozen["run"] = {"experiment_id": "A0", "seed": None, "train_manifest": str(dev_dataset.manifest_path.resolve())}
    (run_dir / "frozen_config.yaml").write_text(yaml.safe_dump(frozen, sort_keys=True), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(environment_metadata(device), indent=2), encoding="utf-8")
    return metrics


def run_ablation(config_path: str | Path, stage: str) -> Path:
    if stage not in {"screen", "final"}:
        raise ValueError("stage must be screen or final")
    config = load_ablation_config(config_path)
    source = ShardedWaymoDataset(_resolve(config, config["cache"]["train_manifest"]))
    stage_config = config[stage]
    train_indices, dev_indices = deterministic_partitions(
        source, int(stage_config["train_targets"]), int(stage_config["dev_targets"]), str(config["split_salt"]),
    )
    train_dataset = ShardedWaymoDataset(source.manifest_path, train_indices)
    dev_dataset = ShardedWaymoDataset(source.manifest_path, dev_indices)
    root = _resolve(config, config["experiment_dir"]) / stage
    root.mkdir(parents=True, exist_ok=True)
    partition = {
        "source_manifest": str(source.manifest_path.resolve()), "split_salt": config["split_salt"],
        "train_indices": train_indices.tolist(), "dev_indices": dev_indices.tolist(),
        "train_scenarios": sorted({sid for _, sid, _ in train_dataset.iter_metadata()}),
        "dev_scenarios": sorted({sid for _, sid, _ in dev_dataset.iter_metadata()}),
    }
    (root / "partition_manifest.json").write_text(json.dumps(partition, indent=2), encoding="utf-8")
    _baseline_run(dev_dataset, root / "A0", config)
    for experiment_id in [f"A{i}" for i in range(1, 8)]:
        for seed in stage_config["seeds"]:
            run_dir = root / experiment_id / f"seed-{seed}"
            complete = all((run_dir / name).is_file() for name in ("checkpoint.pt", "metrics.json", "predictions.npz"))
            if complete:
                print(f"skipping finished run {experiment_id} seed {seed}", flush=True)
            else:
                train_one(experiment_id, int(seed), train_dataset, dev_dataset, run_dir, config)
    aggregate_results(root)
    return root


def _load_run(stage_dir: Path, experiment_id: str, seed: int | None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    directory = stage_dir / experiment_id if seed is None else stage_dir / experiment_id / f"seed-{seed}"
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return metrics, arrays


def _backfill_brier_minfde(directory: Path, metrics: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Add Brier-minFDE to older saved evaluations without rerunning a model."""
    if "metric_brier_minfde" in arrays and "brier_minfde" in metrics.get("overall", {}):
        return metrics, arrays
    required = {"trajectories", "logits", "future", "future_valid", "target_type"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"cannot backfill Brier-minFDE in {directory}: missing {sorted(missing)}")
    summary, per_sample = summarize_predictions(
        arrays["trajectories"], arrays["logits"], arrays["future"], arrays["future_valid"],
        arrays["target_type"], _slices(arrays),
    )
    arrays = dict(arrays)
    arrays["metric_brier_minfde"] = per_sample["brier_minfde"]
    # Preserve run metadata while replacing metric sections with values recomputed from
    # the saved trajectories, logits, and validity masks.
    for key in ("primary_macro_minade", "primary_horizon_seconds", "overall", "per_type", "horizons", "slices", "count"):
        metrics[key] = summary[key]
    metrics_path = directory / "metrics.json"
    _save_predictions(directory / "predictions.npz", arrays)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, arrays


def aggregate_results(stage_dir: str | Path) -> Path:
    stage_dir = Path(stage_dir)
    rows: list[dict[str, Any]] = []
    run_data: dict[tuple[str, int | None], tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    final_label = "8s"
    for experiment_id in EXPERIMENTS:
        directories = [stage_dir / "A0"] if experiment_id == "A0" else sorted((stage_dir / experiment_id).glob("seed-*"))
        for directory in directories:
            seed = None if experiment_id == "A0" else int(directory.name.split("-")[-1])
            metrics, arrays = _load_run(stage_dir, experiment_id, seed)
            metrics, arrays = _backfill_brier_minfde(directory, metrics, arrays)
            run_data[(experiment_id, seed)] = (metrics, arrays)
            final_label = f"{float(metrics.get('primary_horizon_seconds', 8)):g}s"
            primary = metrics.get("primary_macro_minade", metrics.get(f"primary_macro_minade_{final_label}", metrics.get("primary_macro_minade_8s")))
            final_horizon = metrics["horizons"].get(final_label, metrics["horizons"].get("8s", {}))
            rows.append({
                "experiment": experiment_id, "seed": "" if seed is None else seed,
                "macro_minADE_final": primary,
                "top1_ADE": metrics["overall"]["top1_ade"], "top1_FDE": metrics["overall"]["top1_fde"],
                "minADE": metrics["overall"]["minade"], "minFDE": metrics["overall"]["minfde"],
                "Brier-minFDE": metrics["overall"]["brier_minfde"],
                "miss_rate_2m": metrics["overall"]["miss_rate_2m"], "parameters": metrics["parameter_count"],
                "vehicle_minADE": metrics["per_type"].get("vehicle", {}).get("minade", float("nan")),
                "pedestrian_minADE": metrics["per_type"].get("pedestrian", {}).get("minade", float("nan")),
                "cyclist_minADE": metrics["per_type"].get("cyclist", {}).get("minade", float("nan")),
                "minADE_3s": metrics["horizons"].get("3s", {}).get("minade", float("nan")),
                "minADE_5s": metrics["horizons"].get("5s", {}).get("minade", float("nan")),
                "minADE_final": final_horizon.get("minade", float("nan")),
                "turning_minADE": metrics["slices"].get("turning", {}).get("minade", float("nan")),
                "straight_minADE": metrics["slices"].get("straight", {}).get("minade", float("nan")),
                "interactive_minADE": metrics["slices"].get("interactive", {}).get("minade", float("nan")),
                "intersection_minADE": metrics["slices"].get("intersection", {}).get("minade", float("nan")),
            })
    csv_path = stage_dir / "aggregate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reported = ("macro_minADE_final", "top1_ADE", "top1_FDE", "minADE", "minFDE", "Brier-minFDE", "miss_rate_2m", "parameters",
                "vehicle_minADE", "pedestrian_minADE", "cyclist_minADE", "minADE_3s", "minADE_5s", "minADE_final",
                "turning_minADE", "straight_minADE", "interactive_minADE", "intersection_minADE")
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for experiment_id in EXPERIMENTS:
        experiment_rows = [row for row in rows if row["experiment"] == experiment_id]
        summary[experiment_id] = {}
        for metric_name in reported:
            values = np.asarray([float(row[metric_name]) for row in experiment_rows])
            finite = values[np.isfinite(values)]
            summary[experiment_id][metric_name] = {
                "mean": float(finite.mean()) if len(finite) else float("nan"),
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            }

    comparisons: dict[str, Any] = {}
    for name, (left_id, right_id) in COMPARISONS.items():
        seeds = sorted({seed for exp, seed in run_data if exp == left_id} & {seed for exp, seed in run_data if exp == right_id})
        left_values, right_values, target_types = [], [], []
        for seed in seeds:
            left = run_data[(left_id, seed)][1]
            right = run_data[(right_id, seed)][1]
            if not np.array_equal(left["cache_index"], right["cache_index"]):
                raise ValueError(f"unpaired predictions for {left_id} and {right_id}, seed {seed}")
            left_values.append(left["metric_minade"])
            right_values.append(right["metric_minade"])
            target_types.append(left["target_type"])
        comparisons[name] = paired_bootstrap_macro_ci(
            np.concatenate(left_values), np.concatenate(right_values), np.concatenate(target_types),
        ) if seeds else {}
        comparisons[name].update({"left": left_id, "right": right_id, "interpretation": "negative favors the right-hand experiment"})

    report = {"horizon": final_label, "summary": summary, "comparisons_95ci": comparisons, "registered_comparisons": COMPARISONS}
    (stage_dir / "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Motion ablation", "", f"| Experiment | Macro minADE {final_label} (mean +/- std) |", "|---|---:|"]
    for experiment_id in EXPERIMENTS:
        value = summary[experiment_id]["macro_minADE_final"]
        lines.append(f"| {experiment_id} | {value['mean']:.4f} +/- {value['std']:.4f} |")
    lines.extend(["", "## Overall supporting metrics", "", "These are mean +/- std across seeds. Brier-minFDE combines best endpoint error with the probability assigned to that endpoint.", "", "| Experiment | minADE | minFDE | Brier-minFDE | Miss rate @ 2 m | First-ranked ADE | First-ranked FDE | Parameters |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for experiment_id in EXPERIMENTS:
        values = summary[experiment_id]
        names = ("minADE", "minFDE", "Brier-minFDE", "miss_rate_2m", "top1_ADE", "top1_FDE")
        cells = " | ".join(f"{values[name]['mean']:.4f} +/- {values[name]['std']:.4f}" for name in names)
        cells += f" | {values['parameters']['mean']:.0f} |"
        lines.append(f"| {experiment_id} | {cells}")
    lines.extend(["", "## Breakdown minADE", "", f"| Experiment | Vehicle | Pedestrian | Cyclist | 3s | 5s | {final_label} | Turning | Straight | Interactive | Intersection |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for experiment_id in EXPERIMENTS:
        values = summary[experiment_id]
        names = ("vehicle_minADE", "pedestrian_minADE", "cyclist_minADE", "minADE_3s", "minADE_5s", "minADE_final", "turning_minADE", "straight_minADE", "interactive_minADE", "intersection_minADE")
        cells = " | ".join(f"{values[name]['mean']:.4f} +/- {values[name]['std']:.4f}" for name in names)
        lines.append(f"| {experiment_id} | {cells} |")
    lines.extend(["", "## Registered paired comparisons", "", "Differences are right minus left; a wholly negative interval supports an improvement.", "", "| Comparison | Delta | Paired 95% CI |", "|---|---:|---:|"])
    for name, value in comparisons.items():
        lines.append(f"| {name} ({value['left']} -> {value['right']}) | {value.get('mean', float('nan')):.4f} | [{value.get('lower', float('nan')):.4f}, {value.get('upper', float('nan')):.4f}] |")
    report_path = stage_dir / "aggregate.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def load_trained_model(run_dir: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model = build_ablation_model(checkpoint["experiment_id"], checkpoint["history_steps"], checkpoint["future_steps"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
