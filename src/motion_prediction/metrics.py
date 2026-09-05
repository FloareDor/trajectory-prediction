"""Metrics used by trajectory forecasting experiments."""

from __future__ import annotations

import numpy as np


def _as_array(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def ade(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Average displacement error for [..., time, 2] trajectories."""
    error = np.linalg.norm(_as_array(prediction) - _as_array(target), axis=-1)
    if mask is None:
        return float(error.mean())
    valid = np.asarray(mask, dtype=bool)
    return float(error[valid].mean()) if valid.any() else 0.0


def fde(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    prediction_array = _as_array(prediction)
    target_array = _as_array(target)
    if mask is None:
        error = np.linalg.norm(prediction_array[..., -1, :] - target_array[..., -1, :], axis=-1)
        return float(error.mean())
    valid = np.asarray(mask, dtype=bool)
    flat_valid = valid.reshape(-1, valid.shape[-1])
    flat_prediction = prediction_array.reshape(-1, prediction_array.shape[-2], 2)
    flat_target = target_array.reshape(-1, target_array.shape[-2], 2)
    values = []
    for pred, truth, row in zip(flat_prediction, flat_target, flat_valid):
        indices = np.flatnonzero(row)
        if len(indices):
            values.append(np.linalg.norm(pred[indices[-1]] - truth[indices[-1]]))
    return float(np.mean(values)) if values else 0.0


def best_of_k_ade(predictions: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Best ADE across K predictions, with predictions shaped [K, T, 2]."""
    errors = np.linalg.norm(_as_array(predictions) - _as_array(target)[None, ...], axis=-1)
    if mask is None:
        errors = errors.mean(axis=-1)
    else:
        valid = np.asarray(mask, dtype=bool)
        errors = (errors * valid[None]).sum(axis=-1) / max(int(valid.sum()), 1)
    return float(errors.min())


def best_of_k_fde(predictions: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    final = -1 if mask is None or not np.asarray(mask, bool).any() else int(np.flatnonzero(np.asarray(mask, bool))[-1])
    errors = np.linalg.norm(_as_array(predictions)[..., final, :] - _as_array(target)[None, final, :], axis=-1)
    return float(errors.min())
