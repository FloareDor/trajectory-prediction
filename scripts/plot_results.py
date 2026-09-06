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
final_dir = root / "experiments/av2_ablation_60k/final"


def load_run(experiment: str, seed: int) -> dict[str, np.ndarray]:
    """Load one saved dev evaluation."""
    with np.load(final_dir / experiment / f"seed-{seed}" / "predictions.npz", allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def mode_errors(run: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-mode ADE, final error, and endpoint-best mode for a saved run."""
    prediction = run["trajectories"]
    truth = run["future"]
    valid = run["future_valid"].astype(bool)
    distance = np.linalg.norm(prediction - truth[:, None], axis=-1)
    counts = valid.sum(axis=-1).clip(min=1)
    ade = (distance * valid[:, None]).sum(axis=-1) / counts[:, None]
    last_valid = np.array([np.flatnonzero(row)[-1] if row.any() else 0 for row in valid])
    fde = distance[np.arange(len(distance)), :, last_valid]
    return ade, fde, fde.argmin(axis=-1)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


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


# A7: what more ranked guesses buy after the first one.
seeds = (17, 29, 43)
topk_ade, topk_fde = [], []
for seed in seeds:
    run = load_run("A7", seed)
    ade, fde, _ = mode_errors(run)
    ranked = np.argsort(run["logits"], axis=-1)[:, ::-1]
    ranked_ade = np.take_along_axis(ade, ranked, axis=-1)
    ranked_fde = np.take_along_axis(fde, ranked, axis=-1)
    best_ade = np.minimum.accumulate(ranked_ade, axis=-1)
    best_fde = np.minimum.accumulate(ranked_fde, axis=-1)
    type_scores = []
    for type_id in (1, 2, 3):
        chosen = run["target_type"] == type_id
        type_scores.append(best_ade[chosen].mean(axis=0))
    topk_ade.append(np.mean(type_scores, axis=0))
    topk_fde.append(best_fde.mean(axis=0))

x = np.arange(1, 7)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
for ax, values, label in zip(axes, (topk_ade, topk_fde), ("macro minADE (m)", "minFDE (m)")):
    values = np.asarray(values)
    ax.errorbar(x, values.mean(axis=0), yerr=values.std(axis=0, ddof=1), color="#17835c", marker="o", capsize=3)
    ax.set_xticks(x)
    ax.set_xlabel("highest-scored paths kept")
    ax.set_ylabel(label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dde2", linewidth=0.8)
    ax.set_axisbelow(True)
fig.suptitle("A7 gets better coverage when more guesses are allowed", x=0.01, ha="left", weight="bold")
fig.savefig(output / "a7_topk_coverage.png", dpi=180)
plt.close(fig)


# A7: are a mode's scores useful as probabilities that it is the closest endpoint?
probabilities, winners = [], []
for seed in seeds:
    run = load_run("A7", seed)
    _, _, best_endpoint = mode_errors(run)
    probabilities.append(softmax(run["logits"]).ravel())
    winners.append((np.arange(run["logits"].shape[1])[None] == best_endpoint[:, None]).ravel())
probabilities = np.concatenate(probabilities)
winners = np.concatenate(winners)
edges = np.linspace(0, 1, 11)
centers, observed, counts = [], [], []
for start, end in zip(edges[:-1], edges[1:]):
    chosen = (probabilities >= start) & ((probabilities < end) if end < 1 else (probabilities <= end))
    if chosen.any():
        centers.append(probabilities[chosen].mean())
        observed.append(winners[chosen].mean())
        counts.append(chosen.sum())

fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
ax.plot([0, 1], [0, 1], color="#8a8f98", linestyle="--", label="perfect scores")
ax.plot(centers, observed, color="#c16a35", marker="o", label="A7")
for x_value, y_value, count in zip(centers, observed, counts):
    if count >= 500:
        ax.annotate(f"{count / 1000:.0f}k", (x_value, y_value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="score given to a path", ylabel="how often it had the closest endpoint")
ax.set_title("A7 score calibration", loc="left", weight="bold", pad=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#d9dde2", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False)
fig.savefig(output / "a7_calibration.png", dpi=180)
plt.close(fig)


# Which situations benefit from context? Each point is the average paired A3 delta.
slice_labels = ["vehicles", "pedestrians", "cyclists", "turning", "interactive", "intersection"]
context_models = [("A4", "+ neighbors", "#3974a5"), ("A5", "+ map", "#17835c"), ("A6", "+ both", "#c16a35")]
context_scores = {model: [] for model, _, _ in context_models}
for seed in seeds:
    left = load_run("A3", seed)
    if not np.array_equal(left["cache_index"], load_run("A4", seed)["cache_index"]):
        raise ValueError("context runs are not paired")
    selections = [
        left["target_type"] == 1, left["target_type"] == 2, left["target_type"] == 3,
        np.isfinite(left["heading_change"]) & (left["heading_change"] >= np.deg2rad(15)),
        left["interactive"].astype(bool), left["intersection"].astype(bool),
    ]
    for model, _, _ in context_models:
        right = load_run(model, seed)
        delta = right["metric_minade"] - left["metric_minade"]
        context_scores[model].append([delta[selected].mean() for selected in selections])

fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
y = np.arange(len(slice_labels))
for offset, (model, label, color) in zip((-0.22, 0, 0.22), context_models):
    values = np.asarray(context_scores[model])
    ax.errorbar(values.mean(axis=0), y + offset, xerr=values.std(axis=0, ddof=1), fmt="o", color=color, capsize=3, label=label)
ax.axvline(0, color="#8a8f98", linewidth=1)
ax.set_yticks(y, slice_labels)
ax.invert_yaxis()
ax.set_xlabel("change in minADE from A3 (m)")
ax.set_title("where context helps", loc="left", weight="bold", pad=10)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", color="#d9dde2", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="lower left")
fig.savefig(output / "context_by_situation.png", dpi=180)
plt.close(fig)
