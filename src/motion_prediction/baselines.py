"""Simple kinematic baselines."""

from __future__ import annotations

import numpy as np


def constant_position(history: np.ndarray, future_steps: int) -> np.ndarray:
    last = np.asarray(history, dtype=np.float32)[-1]
    return np.repeat(last[None, :], future_steps, axis=0)


def constant_velocity(history: np.ndarray, future_steps: int) -> np.ndarray:
    history = np.asarray(history, dtype=np.float32)
    velocity = history[-1] - history[-2]
    steps = np.arange(1, future_steps + 1, dtype=np.float32)[:, None]
    return history[-1][None, :] + steps * velocity[None, :]


def constant_acceleration(history: np.ndarray, future_steps: int) -> np.ndarray:
    history = np.asarray(history, dtype=np.float32)
    velocity = history[-1] - history[-2]
    acceleration = history[-1] - 2 * history[-2] + history[-3]
    steps = np.arange(1, future_steps + 1, dtype=np.float32)[:, None]
    return history[-1][None, :] + steps * velocity[None, :] + 0.5 * steps * (steps + 1) * acceleration[None, :]
