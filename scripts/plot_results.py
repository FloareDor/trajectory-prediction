"""make the result figures used in the readme."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
results = json.loads((root / "experiments/av2_ablation_60k/final/aggregate.json").read_text())
output = root / "docs/figures"
output.mkdir(parents=True, exist_ok=True)


models = [f"A{i}" for i in range(8)]
names = ["constant\nvelocity", "MLP", "GRU", "target-only\ntransformer",
         "+ neighbors", "+ map", "+ both", "+ six modes"]
means = [results["summary"][model]["macro_minADE_final"]["mean"] for model in models]
stds = [results["summary"][model]["macro_minADE_final"]["std"] for model in models]
colors = ["#8a8f98", "#8a8f98", "#8a8f98", "#3974a5", "#3974a5", "#3974a5", "#3974a5", "#17835c"]

fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
bars = ax.bar(np.arange(8), means, yerr=stds, capsize=3, color=colors, edgecolor="white", linewidth=0.8)
ax.set_xticks(np.arange(8), [f"{model}\n{name}" for model, name in zip(models, names)])
ax.set_ylabel("macro minADE (m)")
ax.set_title("model comparison", loc="left", weight="bold", pad=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#d9dde2", linewidth=0.8)
ax.set_axisbelow(True)
ax.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=4, fontsize=9)
ax.set_ylim(0, max(means) + 0.45)
fig.savefig(output / "model_comparison.png", dpi=180)
plt.close(fig)


keys = ["neighbors_without_map", "map_without_neighbors", "combined_context", "multimodality"]
labels = ["add neighbors\nA3 -> A4", "add map\nA3 -> A5", "add both\nA3 -> A6", "use six modes\nA6 -> A7"]
effects = [results["comparisons_95ci"][key] for key in keys]
means = np.asarray([value["mean"] for value in effects])
lower = means - np.asarray([value["lower"] for value in effects])
upper = np.asarray([value["upper"] for value in effects]) - means

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
y = np.arange(len(keys))
ax.errorbar(means, y, xerr=np.stack([lower, upper]), fmt="o", color="#1f2933", ecolor="#3974a5", capsize=5, markersize=6)
ax.axvline(0, color="#8a8f98", linewidth=1)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("change in macro minADE (m), right model minus left model")
ax.set_title("paired component effects", loc="left", weight="bold", pad=10)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", color="#d9dde2", linewidth=0.8)
ax.set_axisbelow(True)
fig.savefig(output / "component_effects.png", dpi=180)
plt.close(fig)


metric_names = ["minFDE", "Brier-minFDE", "miss_rate_2m"]
metric_labels = ["final minFDE (m)", "Brier-minFDE (m)", "miss rate @ 2 m"]
colors = ["#3974a5", "#17835c", "#c16a35"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
for ax, metric, label, color in zip(axes, metric_names, metric_labels, colors):
    values = [results["summary"][model][metric]["mean"] for model in models]
    bars = ax.bar(np.arange(8), values, color=color, edgecolor="white", linewidth=0.8)
    ax.set_xticks(np.arange(8), models)
    ax.set_ylabel(label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dde2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=8)
fig.suptitle("endpoint and ranking metrics", x=0.01, ha="left", weight="bold")
fig.savefig(output / "endpoint_metrics.png", dpi=180)
plt.close(fig)
