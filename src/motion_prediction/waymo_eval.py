"""small bridge from saved predictions to the official Waymo metric op."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .ablation_data import ShardedWaymoDataset, local_to_global
from .ablation_runner import load_trained_model, predict_dataset


METRIC_CONFIG = """
track_steps_per_second: 10
prediction_steps_per_second: 2
track_history_samples: 10
track_future_samples: 80
step_configurations { measurement_step: 5 lateral_miss_threshold: 1.0 longitudinal_miss_threshold: 2.0 }
step_configurations { measurement_step: 9 lateral_miss_threshold: 1.8 longitudinal_miss_threshold: 3.6 }
step_configurations { measurement_step: 15 lateral_miss_threshold: 3.0 longitudinal_miss_threshold: 6.0 }
max_predictions: 6
speed_scale_lower: 0.5
speed_scale_upper: 1.0
speed_lower_bound: 1.4
speed_upper_bound: 11.0
"""


def prepare_official_inputs(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    local = arrays["trajectories"]
    global_predictions = np.stack([
        local_to_global(local[i], arrays["origin"][i], float(arrays["heading"][i]))
        for i in range(len(local))
    ])
    # waymo evaluates 16 future points at 2 hz: 0.5, 1.0, ..., 8.0 s
    sampled = global_predictions[:, :, 4::5]
    logits = arrays["logits"]
    scores = np.exp(logits - logits.max(axis=-1, keepdims=True))
    scores /= scores.sum(axis=-1, keepdims=True)
    batch = len(sampled)
    return {
        "prediction_trajectory": sampled[:, None, :, None].astype(np.float32),
        "prediction_score": scores[:, None].astype(np.float32),
        "ground_truth_trajectory": arrays["ground_truth"][:, None].astype(np.float32),
        "ground_truth_is_valid": arrays["ground_truth_valid"][:, None].astype(bool),
        "prediction_ground_truth_indices": np.zeros((batch, 1, 1), np.int64),
        "prediction_ground_truth_indices_mask": np.ones((batch, 1, 1), bool),
        "object_type": arrays["target_type"][:, None].astype(np.int64),
        "object_id": arrays["object_id"][:, None].astype(np.int64),
        "scenario_id": arrays["scenario_id"].astype(str),
    }


def run_official_metrics(inputs: dict[str, np.ndarray]) -> dict[str, float]:
    try:
        import tensorflow as tf
        from google.protobuf import text_format
        from waymo_open_dataset.metrics.ops import py_metrics_ops
        from waymo_open_dataset.metrics.python import config_util_py
        from waymo_open_dataset.protos import motion_metrics_pb2
    except ImportError as exc:
        raise RuntimeError("run official evaluation in the pinned Linux/WSL Waymo environment") from exc

    config = text_format.Parse(METRIC_CONFIG, motion_metrics_pb2.MotionMetricsConfig())
    tensors = {key: tf.convert_to_tensor(value) for key, value in inputs.items()}
    values = py_metrics_ops.motion_metrics(config=config.SerializeToString(), **tensors)
    names = config_util_py.get_breakdown_names_from_motion_config(config)
    labels = ("minADE", "minFDE", "MissRate", "OverlapRate", "mAP")
    result = {}
    for label, metric_values in zip(labels, values):
        for name, value in zip(names, np.asarray(metric_values)):
            result[f"{name}/{label}"] = float(value)
    return result


def aggregate_official_metrics(stage_dir: str | Path) -> Path:
    stage_dir = Path(stage_dir)
    rows = []
    for path in sorted(stage_dir.glob("**/official_metrics.json")):
        relative = path.parent.relative_to(stage_dir).parts
        row: dict[str, Any] = {"experiment": relative[0], "seed": relative[1] if len(relative) > 1 else ""}
        row.update(json.loads(path.read_text(encoding="utf-8")))
        rows.append(row)
    if not rows:
        raise ValueError(f"no official metrics found under {stage_dir}")
    fields = ["experiment", "seed"] + sorted({key for row in rows for key in row if key not in {"experiment", "seed"}})
    output = stage_dir / "official_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def evaluate_experiment(run_dir: str | Path, official: bool = True) -> Path:
    run_dir = Path(run_dir)
    if not (run_dir / "frozen_config.yaml").is_file():
        runs = sorted(path.parent for path in run_dir.glob("**/frozen_config.yaml"))
        if not runs:
            raise FileNotFoundError(f"no frozen runs found under {run_dir}")
        outputs = [str(evaluate_experiment(path, official)) for path in runs]
        aggregate = str(aggregate_official_metrics(run_dir)) if official else None
        index = run_dir / "official_evaluation.json"
        index.write_text(json.dumps({"outputs": outputs, "aggregate": aggregate}, indent=2), encoding="utf-8")
        return index
    frozen = yaml.safe_load((run_dir / "frozen_config.yaml").read_text(encoding="utf-8"))
    manifest = Path(frozen["cache"]["validation_manifest"])
    if not manifest.is_absolute():
        manifest = Path.cwd() / manifest
    dataset = ShardedWaymoDataset(manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_id = frozen["run"]["experiment_id"]
    if experiment_id == "A0":
        model = None
    else:
        model, _ = load_trained_model(run_dir, device)
    _, arrays = predict_dataset(model, dataset, device, int(frozen["training"]["batch_size"]), experiment_id)
    output = run_dir / "official_predictions.npz"
    with output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    if official:
        metrics = run_official_metrics(prepare_official_inputs(arrays))
        metrics_path = run_dir / "official_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics_path
    return output
