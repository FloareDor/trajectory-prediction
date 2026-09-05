"""plot one real scene from the finished experiment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from motion_prediction.ablation_data import ShardedWaymoDataset


root = Path(__file__).resolve().parents[1]
run = root / "experiments/av2_ablation_60k/final"
cache = root / "data/processed/av2_60k/train-manifest.json"
output = root / "docs/figures/trajectory_example.png"


def actor_box(axis, values, color, label=None):
    x, y = values[:2]
    heading = np.arctan2(values[4], values[5])
    length, width = values[6:8]
    corners = np.array([
        [length / 2, width / 2], [length / 2, -width / 2],
        [-length / 2, -width / 2], [-length / 2, width / 2],
    ])
    c, s = np.cos(heading), np.sin(heading)
    corners = corners @ np.array([[c, s], [-s, c]]) + [x, y]
    axis.add_patch(Polygon(corners, closed=True, color=color, zorder=5, label=label))


def draw_scene(axis, sample, show_labels=True):
    for line, valid, kind in zip(sample["map_polylines"], sample["map_valid"], sample["map_types"]):
        points = line[valid, :2]
        if len(points) < 2:
            continue
        color = "#d4a72c" if kind == 5 else "#c3c9d1"
        width = 1.5 if kind in {3, 5, 15} else 0.9
        axis.plot(points[:, 0], points[:, 1], color=color, linewidth=width, zorder=0)

    neighbor_label = True
    for track, valid in zip(sample["neighbors"], sample["neighbor_valid"]):
        points = track[valid]
        if not len(points):
            continue
        axis.plot(points[:, 0], points[:, 1], color="#8b949e", linewidth=1.1, alpha=0.75, zorder=2)
        actor_box(axis, points[-1], "#8b949e", "nearby agents" if show_labels and neighbor_label else None)
        neighbor_label = False

    history = sample["target_history"][sample["history_valid"]]
    axis.plot(history[:, 0], history[:, 1], color="#1677b8", linewidth=3,
              label="target history" if show_labels else None, zorder=3)
    actor_box(axis, history[-1], "#1677b8", "target now" if show_labels else None)
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("#f7f8fa")
    axis.grid(False)
    axis.set_xlabel("metres")
    axis.set_ylabel("metres")
    axis.spines[["top", "right"]].set_visible(False)


with np.load(run / "A6/seed-17/predictions.npz") as file:
    a6 = {key: file[key] for key in file.files}
with np.load(run / "A7/seed-17/predictions.npz") as file:
    a7 = {key: file[key] for key in file.files}

gain = a6["metric_minade"] - a7["metric_minade"]
turning = np.abs(a7["heading_change"]) >= np.deg2rad(15)
choices = a7["interactive"] & turning
row = int(np.argmax(np.where(choices, gain, -np.inf)))

dataset = ShardedWaymoDataset(cache)
raw = dataset[int(a7["cache_index"][row])]
sample = {key: value.numpy() if hasattr(value, "numpy") else value for key, value in raw.items()}

truth = a7["future"][row][a7["future_valid"][row]]
single = a6["trajectories"][row, 0]
modes = a7["trajectories"][row]
scores = a7["logits"][row]
top_mode = int(np.argmax(scores))
errors = np.linalg.norm(modes[:, :len(truth)] - truth[None], axis=-1).mean(axis=1)
best_mode = int(np.argmin(errors))

all_points = [sample["target_history"][sample["history_valid"], :2], truth, single, modes.reshape(-1, 2)]
for track, valid in zip(sample["neighbors"], sample["neighbor_valid"]):
    if valid.any():
        all_points.append(track[valid, :2])
points = np.concatenate(all_points)
low, high = points.min(axis=0), points.max(axis=0)
padding = max(8.0, float(np.max(high - low)) * 0.12)
limits = low - padding, high + padding

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
draw_scene(axes[0], sample)
axes[0].plot(truth[:, 0], truth[:, 1], color="#111827", linewidth=3, label="real future", zorder=4)
axes[0].set_title("scene + ground truth", loc="left", weight="bold")
axes[0].legend(loc="best", frameon=True, fontsize=9)

draw_scene(axes[1], sample, show_labels=False)
axes[1].plot(truth[:, 0], truth[:, 1], color="#111827", linewidth=3, label="real future", zorder=4)
axes[1].plot(single[:, 0], single[:, 1], color="#d1495b", linewidth=2.2, label="A6: one path", zorder=3)
other_label = True
for index, path in enumerate(modes):
    if index in {top_mode, best_mode}:
        continue
    axes[1].plot(path[:, 0], path[:, 1], color="#b79ced", linewidth=1.3, alpha=0.65,
                 label="other A7 paths" if other_label else None, zorder=2)
    other_label = False
axes[1].plot(modes[top_mode, :, 0], modes[top_mode, :, 1], color="#7b2cbf", linewidth=2.4,
             label="A7 top choice", zorder=3)
if best_mode != top_mode:
    axes[1].plot(modes[best_mode, :, 0], modes[best_mode, :, 1], color="#16865b", linewidth=2.6,
                 label="A7 best path (minADE)", zorder=3)
axes[1].set_title("model predictions", loc="left", weight="bold")
axes[1].legend(loc="best", frameon=True, fontsize=9)

for axis in axes:
    axis.set_xlim(limits[0][0], limits[1][0])
    axis.set_ylim(limits[0][1], limits[1][1])

fig.savefig(output, dpi=180)
plt.close(fig)
print(f"wrote {output}")
