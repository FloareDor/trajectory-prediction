"""Bird's-eye-view rendering utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import TrajectorySample


def _scene_limits(sample: TrajectorySample, prediction: np.ndarray | None, baseline: np.ndarray | None) -> tuple[float, float, float, float]:
    points = [sample.history, sample.future]
    for neighbor, mask in zip(sample.neighbors, sample.neighbor_mask):
        if mask.any():
            points.append(neighbor[mask])
    if prediction is not None:
        points.append(np.asarray(prediction).reshape(-1, 2))
    if baseline is not None:
        points.append(np.asarray(baseline).reshape(-1, 2))
    combined = np.concatenate(points, axis=0)
    lower = combined.min(axis=0)
    upper = combined.max(axis=0)
    padding = max(12.0, float(np.max(upper - lower)) * 0.12)
    return lower[0] - padding, upper[0] + padding, lower[1] - padding, upper[1] + padding


def _visible_polyline(points: np.ndarray, limits: tuple[float, float, float, float]) -> np.ndarray:
    """Keep only geometry near the prediction window to avoid a giant empty map."""
    xmin, xmax, ymin, ymax = limits
    padding = 8.0
    visible = (points[:, 0] >= xmin - padding) & (points[:, 0] <= xmax + padding) & (points[:, 1] >= ymin - padding) & (points[:, 1] <= ymax + padding)
    return points[visible]


def _draw_actor_box(
    axis: plt.Axes,
    center: np.ndarray,
    heading: float,
    length: float,
    width: float,
    color: str,
    label: str | None = None,
) -> None:
    """Draw an oriented top-down vehicle footprint instead of a generic dot."""
    corners = np.array(
        [[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]],
        dtype=np.float32,
    )
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = corners @ rotation.T + center
    axis.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.92, linewidth=0, label=label, zorder=6)
    # A short nose line makes vehicle orientation understandable at a glance.
    nose = center + np.array([np.cos(heading), np.sin(heading)], dtype=np.float32) * length * 0.36
    axis.plot([center[0], nose[0]], [center[1], nose[1]], color="white", linewidth=1.1, zorder=7)


def _traffic_light_style(state: int) -> tuple[str, str]:
    # Values follow Waymo's TrafficSignalLaneState.State enum.
    if state in {1, 4, 7}:
        return "#dc2626", "R"
    if state in {2, 5, 8}:
        return "#f59e0b", "Y"
    if state in {3, 6}:
        return "#16a34a", "G"
    return "#64748b", "?"


def _draw_scene(axis: plt.Axes, sample: TrajectorySample, limits: tuple[float, float, float, float]) -> None:
    axis.set_facecolor("#6f805b")  # neutral land cover; the road geometry comes from the actual vector map.
    for polygon, mask in zip(sample.road_polygons, sample.road_polygon_mask):
        # Let Matplotlib clip full lane polygons to the selected scene window;
        # dropping exterior vertices would turn a long road into a broken fill.
        valid = polygon[mask]
        if len(valid) > 2:
            axis.fill(valid[:, 0], valid[:, 1], color="#343a40", alpha=1.0, linewidth=0, zorder=0)
    map_colors = {
        "lane_center": "#94a3b8",
        "road_line": "#f8fafc",
        "road_edge": "#111827",
        "crosswalk": "#f8fafc",
        "speed_bump": "#f59e0b",
        "stop_sign": "#dc2626",
    }
    stop_label_drawn = False
    for polyline, mask, polyline_type in zip(sample.map_polylines, sample.map_mask, sample.map_types):
        valid = _visible_polyline(polyline[mask], limits)
        if len(valid) > 1:
            if polyline_type == "crosswalk" and len(valid) > 2:
                axis.fill(valid[:, 0], valid[:, 1], facecolor="#f8fafc", edgecolor="#f8fafc", alpha=0.32, hatch="///", linewidth=0.6, zorder=1)
                continue
            if polyline_type == "speed_bump" and len(valid) > 2:
                axis.fill(valid[:, 0], valid[:, 1], color=map_colors[polyline_type], alpha=0.42, linewidth=0, zorder=1)
            color = map_colors.get(polyline_type, "#d6dbe4")
            linestyle = "-"
            linewidth = 1.3
            if polyline_type == "lane_center":
                linestyle, linewidth = (0, (4, 4)), 0.9
            elif "broken" in polyline_type:
                linestyle, linewidth = (0, (7, 5)), 1.5
            elif "yellow" in polyline_type:
                color, linewidth = "#fbbf24", 1.8
            elif polyline_type == "road_edge":
                linewidth = 2.1
            axis.plot(valid[:, 0], valid[:, 1], color=color, linestyle=linestyle, linewidth=linewidth, zorder=2)
        elif len(valid) == 1 and polyline_type == "stop_sign":
            axis.scatter(valid[0, 0], valid[0, 1], marker="h", s=115, color=map_colors[polyline_type], edgecolor="#111827", linewidth=0.8, label="stop sign" if not stop_label_drawn else None, zorder=8)
            axis.text(valid[0, 0], valid[0, 1], "STOP", color="white", ha="center", va="center", fontsize=5.5, fontweight="bold", zorder=9)
            stop_label_drawn = True
    ego_label_drawn = False
    for index, (neighbor, mask) in enumerate(zip(sample.neighbors, sample.neighbor_mask)):
        valid = neighbor[mask]
        if len(valid):
            axis.plot(valid[:, 0], valid[:, 1], color="#cbd5e1", alpha=0.62, linewidth=1.3, zorder=3)
            is_ego = bool(sample.neighbor_is_ego[index])
            _draw_actor_box(
                axis,
                valid[-1],
                float(sample.neighbor_headings[index]),
                float(sample.neighbor_lengths[index]),
                float(sample.neighbor_widths[index]),
                "#fb923c" if is_ego else "#94a3b8",
                "ego / SDC" if is_ego and not ego_label_drawn else None,
            )
            ego_label_drawn = ego_label_drawn or is_ego
    axis.plot(sample.history[:, 0], sample.history[:, 1], color="#60a5fa", linewidth=2.6, label="target history", zorder=5)
    _draw_actor_box(
        axis,
        sample.history[-1],
        sample.target_heading,
        sample.target_length,
        sample.target_width,
        "#38bdf8" if not sample.target_is_ego else "#fb923c",
        "target vehicle" if not sample.target_is_ego else "target / ego",
    )
    signal_label_drawn = False
    for position, state in zip(sample.traffic_light_positions, sample.traffic_light_states):
        if state == 0:
            continue
        color, state_label = _traffic_light_style(int(state))
        axis.scatter(position[0], position[1], s=115, color=color, edgecolor="#111827", linewidth=0.8, marker="o", label="traffic signal" if not signal_label_drawn else None, zorder=8)
        axis.text(position[0], position[1], state_label, color="white", ha="center", va="center", fontsize=8, fontweight="bold", zorder=9)
        signal_label_drawn = True
    axis.plot(sample.future[:, 0], sample.future[:, 1], color="#f8fafc", linewidth=2.7, label="ground truth", zorder=5)


def plot_scene(sample: TrajectorySample, output_path: str | Path) -> None:
    """Render a bird's-eye-view scene before any model is involved."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limits = _scene_limits(sample, None, None)
    fig, axis = plt.subplots(figsize=(9, 7))
    _draw_scene(axis, sample, limits)
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"Bird's-eye view — {sample.scenario_id}, target {sample.target_id}", color="#111827")
    axis.legend(loc="upper left", frameon=True, framealpha=0.92)
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_prediction(
    sample: TrajectorySample,
    prediction: np.ndarray,
    output_path: str | Path,
    baseline: np.ndarray | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limits = _scene_limits(sample, prediction, baseline)
    fig, axis = plt.subplots(figsize=(9, 7))
    _draw_scene(axis, sample, limits)
    prediction = np.asarray(prediction)
    if prediction.ndim == 2:
        prediction = prediction[None, ...]
    for index, path in enumerate(prediction):
        color = "#c084fc" if index == 0 else "#e879f9"
        label = "model prediction" if index == 0 else f"candidate {index + 1}"
        axis.plot(path[:, 0], path[:, 1], "--", color=color, linewidth=2.3 if index == 0 else 1.6, alpha=1.0 if index == 0 else 0.72, label=label, zorder=4)
    if baseline is not None:
        axis.plot(baseline[:, 0], baseline[:, 1], ":", color="#fbbf24", linewidth=2.2, label="constant velocity", zorder=4)
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"Prediction — {sample.scenario_id}, target {sample.target_id}", color="#111827")
    axis.legend(loc="upper left", frameon=True, framealpha=0.92)
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
