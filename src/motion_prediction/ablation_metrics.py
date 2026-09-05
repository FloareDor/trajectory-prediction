"""Fast masked metrics, slices, and paired uncertainty estimates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}


def per_sample_metrics(
    trajectories: np.ndarray,
    logits: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    horizon: int | None = None,
) -> dict[str, np.ndarray]:
    """Return top-1 and best-of-K errors for [N,K,T,2] predictions."""
    pred = np.asarray(trajectories, np.float32)
    truth = np.asarray(target, np.float32)
    mask = np.asarray(valid, bool)
    if horizon is not None:
        pred, truth, mask = pred[:, :, :horizon], truth[:, :horizon], mask[:, :horizon]
    distance = np.linalg.norm(pred - truth[:, None], axis=-1)
    denominator = mask.sum(axis=-1).clip(min=1)
    mode_ade = (distance * mask[:, None]).sum(axis=-1) / denominator[:, None]
    final_indices = np.maximum(mask.sum(axis=-1) - 1, 0)
    # Validity can contain internal gaps; find the actual final valid index.
    for row, row_mask in enumerate(mask):
        indices = np.flatnonzero(row_mask)
        final_indices[row] = indices[-1] if len(indices) else 0
    mode_fde = np.stack([distance[index, :, final_indices[index]] for index in range(len(pred))])
    raw_logits = np.asarray(logits, np.float64)
    top = raw_logits.argmax(axis=-1)
    rows = np.arange(len(pred))
    best_endpoint = mode_fde.argmin(axis=-1)
    # Subtracting the row maximum keeps the normalization stable for large logits.
    shifted = raw_logits - raw_logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    minfde = mode_fde[rows, best_endpoint]
    brier_minfde = minfde + (1.0 - probabilities[rows, best_endpoint]) ** 2
    # With one possible path there is no ranking decision to score.
    if pred.shape[1] == 1:
        brier_minfde = minfde.copy()
    return {
        "top1_ade": mode_ade[rows, top],
        "top1_fde": mode_fde[rows, top],
        "minade": mode_ade.min(axis=-1),
        "minfde": minfde,
        "brier_minfde": brier_minfde,
        "miss_rate_2m": (mode_fde.min(axis=-1) > 2.0).astype(np.float32),
        "has_valid_future": mask.any(axis=-1),
    }


def _mean_valid(values: np.ndarray, selection: np.ndarray) -> float:
    chosen = values[selection & np.isfinite(values)]
    return float(chosen.mean()) if len(chosen) else float("nan")


def summarize_predictions(
    trajectories: np.ndarray,
    logits: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    target_types: np.ndarray,
    slices: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    arrays = per_sample_metrics(trajectories, logits, target, valid)
    usable = arrays["has_valid_future"].astype(bool)
    overall = {key: _mean_valid(value, usable) for key, value in arrays.items() if key != "has_valid_future"}
    per_type: dict[str, dict[str, float]] = {}
    for type_id, type_name in TYPE_NAMES.items():
        selection = usable & (target_types == type_id)
        if selection.any():
            per_type[type_name] = {key: _mean_valid(value, selection) for key, value in arrays.items() if key != "has_valid_future"}
    type_values = [entry["minade"] for name, entry in per_type.items() if name in {"vehicle", "pedestrian", "cyclist"}]
    macro_minade = float(np.mean(type_values)) if type_values else float("nan")
    horizons: dict[str, dict[str, float]] = {}
    choices = [(seconds, seconds * 10) for seconds in (3, 5, 8) if seconds * 10 <= target.shape[1]]
    final_seconds = target.shape[1] / 10
    if not choices or choices[-1][1] != target.shape[1]:
        choices.append((final_seconds, target.shape[1]))
    for seconds, steps in choices:
        values = per_sample_metrics(trajectories, logits, target, valid, steps)
        label = f"{seconds:g}s"
        horizons[label] = {key: _mean_valid(value, values["has_valid_future"].astype(bool)) for key, value in values.items() if key != "has_valid_future"}
    slice_metrics: dict[str, dict[str, float]] = {}
    for name, raw_selection in (slices or {}).items():
        selection = usable & np.asarray(raw_selection, bool)
        slice_metrics[name] = {key: _mean_valid(value, selection) for key, value in arrays.items() if key != "has_valid_future"}
        slice_metrics[name]["count"] = int(selection.sum())
    summary: dict[str, object] = {
        "primary_macro_minade": macro_minade,
        f"primary_macro_minade_{final_seconds:g}s": macro_minade,
        "primary_horizon_seconds": final_seconds,
        "overall": overall,
        "per_type": per_type,
        "horizons": horizons,
        "slices": slice_metrics,
        "count": int(usable.sum()),
    }
    return summary, arrays


def paired_bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 2027,
) -> dict[str, float]:
    """CI for paired ``right - left`` metric differences."""
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    delta = right[valid] - left[valid]
    if not len(delta):
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "pairs": 0}
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        bootstrap[start : start + count] = delta[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(delta.mean()),
        "lower": float(np.quantile(bootstrap, alpha)),
        "upper": float(np.quantile(bootstrap, 1.0 - alpha)),
        "pairs": int(len(delta)),
    }


def paired_bootstrap_macro_ci(
    left: np.ndarray,
    right: np.ndarray,
    target_types: np.ndarray,
    samples: int = 10_000,
    seed: int = 2027,
) -> dict[str, float]:
    """paired bootstrap with equal weight for vehicle, pedestrian and cyclist."""
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    target_types = np.asarray(target_types)
    groups = []
    for type_id in (1, 2, 3):
        keep = (target_types == type_id) & np.isfinite(left) & np.isfinite(right)
        if keep.any():
            groups.append((right[keep] - left[keep]).copy())
    if not groups:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "pairs": 0}
    rng = np.random.default_rng(seed)
    values = np.zeros(samples, np.float64)
    for group in groups:
        for start in range(0, samples, 1000):
            count = min(1000, samples - start)
            indices = rng.integers(0, len(group), size=(count, len(group)))
            values[start : start + count] += group[indices].mean(axis=1) / len(groups)
    delta = float(np.mean([group.mean() for group in groups]))
    return {
        "mean": delta, "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)), "pairs": int(sum(map(len, groups))),
    }
