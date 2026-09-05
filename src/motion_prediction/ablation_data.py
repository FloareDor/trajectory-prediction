"""Versioned, sharded data format for the Waymo ablation study.

The original :mod:`motion_prediction.data` structures remain useful for
visualisation.  This module owns the fixed-shape, mask-preserving tensors used
by the reproducible training pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import AgentTrack, Scenario


CACHE_VERSION = 2
AGENT_TYPE_IDS = {"0": 0, "TYPE_UNSET": 0, "1": 1, "TYPE_VEHICLE": 1,
                  "2": 2, "TYPE_PEDESTRIAN": 2, "3": 3, "TYPE_CYCLIST": 3,
                  "4": 4, "TYPE_OTHER": 4, "vehicle": 1, "pedestrian": 2,
                  "cyclist": 3, "other": 4}
MAP_TYPE_NAMES = (
    "unknown", "lane_center", "road_line", "road_edge", "stop_sign",
    "crosswalk", "speed_bump", "road_line_broken_single_white",
    "road_line_solid_single_white", "road_line_solid_double_white",
    "road_line_broken_single_yellow", "road_line_broken_double_yellow",
    "road_line_solid_single_yellow", "road_line_solid_double_yellow",
    "road_line_passing_double_yellow", "controlled_lane",
)
MAP_TYPE_IDS = {name: index for index, name in enumerate(MAP_TYPE_NAMES)}


@dataclass(frozen=True)
class CacheSpec:
    history_steps: int = 11
    future_steps: int = 80
    max_neighbors: int = 8
    max_polylines: int = 32
    polyline_points: int = 32
    timestep_seconds: float = 0.1


def agent_type_id(value: str | int) -> int:
    return AGENT_TYPE_IDS.get(str(value), 4)


def map_type_id(value: str) -> int:
    return MAP_TYPE_IDS.get(value, 0)


def rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    value = np.asarray(angle, dtype=np.float32)
    return (value + np.pi) % (2 * np.pi) - np.pi


def finite_difference_velocity(positions: np.ndarray, valid: np.ndarray, dt: float) -> np.ndarray:
    """Masked finite differences without bridging gaps in a trajectory."""
    positions = np.asarray(positions, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    velocity = np.zeros_like(positions)
    adjacent = valid[1:] & valid[:-1]
    velocity[1:][adjacent] = (positions[1:][adjacent] - positions[:-1][adjacent]) / dt
    # use a forward difference at the start or after a gap
    forward = valid[:-1] & valid[1:]
    candidate = (positions[1:] - positions[:-1]) / dt
    missing = valid[:-1] & (np.linalg.norm(velocity[:-1], axis=-1) == 0) & forward
    velocity[:-1][missing] = candidate[missing]
    return velocity


def _track_velocities(track: AgentTrack, dt: float) -> np.ndarray:
    if track.velocities is not None:
        return np.asarray(track.velocities, dtype=np.float32)
    return finite_difference_velocity(track.positions, track.valid, dt)


def _step_sizes(track: AgentTrack) -> tuple[np.ndarray, np.ndarray]:
    count = len(track.positions)
    lengths = np.asarray(track.lengths, dtype=np.float32) if track.lengths is not None else np.full(count, track.length, np.float32)
    widths = np.asarray(track.widths, dtype=np.float32) if track.widths is not None else np.full(count, track.width, np.float32)
    return lengths, widths


def scenario_target_record(scenario: Scenario, target_id: int, spec: CacheSpec = CacheSpec()) -> dict[str, np.ndarray]:
    """Create one fixed-shape record while retaining incomplete trajectories."""
    target = next((a for a in scenario.agents if a.object_id == target_id), None)
    if target is None:
        raise ValueError(f"target {target_id} is missing from {scenario.scenario_id}")
    current = scenario.current_time_index if scenario.current_time_index is not None else spec.history_steps - 1
    if current >= len(target.positions) or not target.valid[current]:
        raise ValueError(f"target {target_id} is invalid at the current frame")

    origin = np.asarray(target.positions[current], dtype=np.float32)
    global_heading = float(target.heading[current]) if target.heading is not None else 0.0
    to_local = rotation_matrix(-global_heading)
    history_indices = np.arange(current - spec.history_steps + 1, current + 1)
    future_indices = np.arange(current + 1, current + spec.future_steps + 1)

    def encode_track(track: AgentTrack, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.zeros((len(indices), 8), dtype=np.float32)
        in_range = (indices >= 0) & (indices < len(track.positions))
        safe = indices[in_range]
        mask = np.zeros(len(indices), dtype=bool)
        if not len(safe):
            return features, mask
        mask[in_range] = track.valid[safe]
        velocity = _track_velocities(track, scenario.timestep_seconds)[safe] @ to_local.T
        positions = (track.positions[safe] - origin) @ to_local.T
        heading = np.zeros(len(safe), dtype=np.float32)
        if track.heading is not None:
            heading = wrap_angle(np.asarray(track.heading)[safe] - global_heading)
        lengths, widths = _step_sizes(track)
        encoded = np.stack(
            [positions[:, 0], positions[:, 1], velocity[:, 0], velocity[:, 1],
             np.sin(heading), np.cos(heading), lengths[safe], widths[safe]], axis=-1,
        ).astype(np.float32)
        features[in_range] = encoded
        features[~mask] = 0.0
        return features, mask

    target_history, history_valid = encode_track(target, history_indices)
    future = np.zeros((spec.future_steps, 2), dtype=np.float32)
    future_valid = np.zeros(spec.future_steps, dtype=bool)
    future_in_range = (future_indices >= 0) & (future_indices < len(target.positions))
    future_safe = future_indices[future_in_range]
    future_valid[future_in_range] = target.valid[future_safe]
    future[future_in_range] = (target.positions[future_safe] - origin) @ to_local.T
    future[~future_valid] = 0.0

    neighbors = np.zeros((spec.max_neighbors, spec.history_steps, 8), dtype=np.float32)
    neighbor_valid = np.zeros((spec.max_neighbors, spec.history_steps), dtype=bool)
    neighbor_types = np.zeros(spec.max_neighbors, dtype=np.int64)
    candidates = [a for a in scenario.agents if a.object_id != target_id and current < len(a.positions) and a.valid[current]]
    candidates.sort(key=lambda a: (float(np.linalg.norm(a.positions[current] - origin)), int(a.object_id)))
    for index, agent in enumerate(candidates[:spec.max_neighbors]):
        neighbors[index], neighbor_valid[index] = encode_track(agent, history_indices)
        neighbor_types[index] = agent_type_id(agent.object_type)

    map_features = np.zeros((spec.max_polylines, spec.polyline_points, 4), dtype=np.float32)
    map_valid = np.zeros((spec.max_polylines, spec.polyline_points), dtype=bool)
    map_types = np.zeros(spec.max_polylines, dtype=np.int64)
    polylines = sorted(
        (polyline for polyline in scenario.map_polylines if len(polyline.points)),
        key=lambda p: (float(np.linalg.norm(np.asarray(p.points) - origin, axis=-1).min()), p.polyline_type),
    )
    for poly_index, polyline in enumerate(polylines[:spec.max_polylines]):
        if not len(polyline.points):
            continue
        indices = np.linspace(0, len(polyline.points) - 1, min(spec.polyline_points, len(polyline.points))).round().astype(int)
        points = (np.asarray(polyline.points, dtype=np.float32)[indices] - origin) @ to_local.T
        direction = np.zeros_like(points)
        if len(points) > 1:
            direction[:-1] = points[1:] - points[:-1]
            direction[-1] = direction[-2]
            norm = np.linalg.norm(direction, axis=-1, keepdims=True)
            direction = np.divide(direction, norm, out=np.zeros_like(direction), where=norm > 1e-6)
        count = len(points)
        map_features[poly_index, :count, :2] = points
        map_features[poly_index, :count, 2:] = direction
        map_valid[poly_index, :count] = True
        map_types[poly_index] = map_type_id(polyline.polyline_type)

    gt_steps = spec.history_steps + spec.future_steps
    gt = np.zeros((gt_steps, 7), dtype=np.float32)
    gt_valid = np.zeros(gt_steps, dtype=bool)
    gt_indices = np.concatenate([history_indices, future_indices])
    gt_in_range = (gt_indices >= 0) & (gt_indices < len(target.positions))
    gt_safe = gt_indices[gt_in_range]
    lengths, widths = _step_sizes(target)
    headings = np.asarray(target.heading, dtype=np.float32) if target.heading is not None else np.zeros(len(target.positions), np.float32)
    velocities = _track_velocities(target, scenario.timestep_seconds)
    gt[gt_in_range] = np.stack(
        [target.positions[gt_safe, 0], target.positions[gt_safe, 1], lengths[gt_safe], widths[gt_safe],
         headings[gt_safe], velocities[gt_safe, 0], velocities[gt_safe, 1]], axis=-1,
    )
    gt_valid[gt_in_range] = target.valid[gt_safe]
    gt[~gt_valid] = 0.0

    interaction = False
    target_future_valid = target.valid[future_safe]
    for agent in scenario.agents:
        if agent.object_id == target_id:
            continue
        common = future_safe < len(agent.positions)
        if not common.any():
            continue
        idx = future_safe[common]
        valid = target.valid[idx] & agent.valid[idx]
        if valid.any() and np.any(np.linalg.norm(target.positions[idx][valid] - agent.positions[idx][valid], axis=-1) < 10.0):
            interaction = True
            break
    valid_future_headings = headings[future_safe][target_future_valid] if len(future_safe) else np.asarray([])
    heading_change = float(abs(wrap_angle(valid_future_headings[-1] - global_heading))) if len(valid_future_headings) else np.nan
    intersection_types = {"crosswalk", "stop_sign"}
    intersection = any(
        p.polyline_type in intersection_types and len(p.points) and np.linalg.norm(np.asarray(p.points) - origin, axis=-1).min() <= 30.0
        for p in scenario.map_polylines
    ) or any(
        p.polyline_type == "controlled_lane" and len(p.points) and np.linalg.norm(np.asarray(p.points) - origin, axis=-1).min() <= 30.0
        for p in scenario.map_polylines
    )

    return {
        "target_history": target_history,
        "history_valid": history_valid,
        "neighbors": neighbors,
        "neighbor_valid": neighbor_valid,
        "neighbor_types": neighbor_types,
        "target_type": np.asarray(agent_type_id(target.object_type), dtype=np.int64),
        "map_polylines": map_features,
        "map_valid": map_valid,
        "map_types": map_types,
        "future": future,
        "future_valid": future_valid,
        "origin": origin,
        "heading": np.asarray(global_heading, dtype=np.float32),
        "scenario_id": np.asarray(scenario.scenario_id),
        "object_id": np.asarray(target.object_id, dtype=np.int64),
        "ground_truth": gt,
        "ground_truth_valid": gt_valid,
        "heading_change": np.asarray(heading_change, dtype=np.float32),
        "interactive": np.asarray(interaction, dtype=bool),
        "intersection": np.asarray(intersection, dtype=bool),
    }


class ShardedCacheWriter:
    """Incrementally write compressed NPZ shards and a checksummed manifest."""

    def __init__(self, output_dir: str | Path, split: str, spec: CacheSpec = CacheSpec(), shard_size: int = 1024):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.spec = spec
        self.shard_size = shard_size
        self.pending: list[Mapping[str, np.ndarray]] = []
        self.shards: list[dict[str, object]] = []
        self.count = 0

    def add(self, record: Mapping[str, np.ndarray]) -> None:
        self.pending.append(record)
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        name = f"{self.split}-{len(self.shards):05d}.npz"
        path = self.root / name
        keys = self.pending[0].keys()
        payload = {key: np.stack([record[key] for record in self.pending]) for key in keys}
        with path.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        digest = sha256(path.read_bytes()).hexdigest()
        scenarios = np.unique(payload["scenario_id"]).tolist()
        self.shards.append({"path": name, "count": len(self.pending), "sha256": digest, "scenarios": len(scenarios)})
        self.count += len(self.pending)
        self.pending.clear()

    def close(self, stats: Mapping[str, object] | None = None) -> Path:
        self.flush()
        manifest = {
            "format": "waymo-motion-vector-cache",
            "version": CACHE_VERSION,
            "split": self.split,
            "count": self.count,
            "spec": asdict(self.spec),
            "shards": self.shards,
            "stats": dict(stats or {}),
        }
        path = self.root / f"{self.split}-manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


def read_manifest(path: str | Path, verify: bool = False) -> dict[str, object]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != CACHE_VERSION:
        raise ValueError(f"cache {path} has version {manifest.get('version')}; version {CACHE_VERSION} is required (regenerate it)")
    if verify:
        for shard in manifest["shards"]:
            shard_path = path.parent / shard["path"]
            if sha256(shard_path.read_bytes()).hexdigest() != shard["sha256"]:
                raise ValueError(f"checksum mismatch: {shard_path}")
    return manifest


class ShardedWaymoDataset(Dataset):
    """Lazy random-access dataset over a v2 manifest."""

    def __init__(self, manifest_path: str | Path, indices: Iterable[int] | None = None):
        self.manifest_path = Path(manifest_path)
        self.manifest = read_manifest(self.manifest_path)
        self.shards = list(self.manifest["shards"])
        counts = np.asarray([int(s["count"]) for s in self.shards], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(counts)])
        self.indices = np.arange(int(self.offsets[-1]), dtype=np.int64) if indices is None else np.asarray(list(indices), dtype=np.int64)
        self._loaded_shards: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def _record(self, absolute_index: int) -> dict[str, np.ndarray]:
        shard_index = int(np.searchsorted(self.offsets, absolute_index, side="right") - 1)
        local_index = absolute_index - int(self.offsets[shard_index])
        if shard_index not in self._loaded_shards:
            with np.load(self.manifest_path.parent / self.shards[shard_index]["path"], allow_pickle=False) as data:
                self._loaded_shards[shard_index] = {key: data[key] for key in data.files}
        return {key: value[local_index] for key, value in self._loaded_shards[shard_index].items()}

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self._record(int(self.indices[index]))
        output: dict[str, object] = {}
        for key, value in record.items():
            if value.dtype.kind in "USO":
                output[key] = str(value)
            else:
                output[key] = torch.from_numpy(np.asarray(value))
        output["cache_index"] = int(self.indices[index])
        return output

    def iter_metadata(self) -> Iterator[tuple[int, str, int]]:
        for index in self.indices:
            record = self._record(int(index))
            yield int(index), str(record["scenario_id"]), int(record["target_type"])


def deterministic_partitions(
    dataset: ShardedWaymoDataset,
    train_targets: int,
    dev_targets: int,
    salt: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build disjoint scenario-level partitions in stable hash order.

    Whole scenarios are admitted only when they fit the requested target
    budget, so a scenario can never be split merely to hit an exact count.
    """
    scenarios: dict[str, list[int]] = {}
    for index, scenario_id, _ in dataset.iter_metadata():
        scenarios.setdefault(scenario_id, []).append(index)
    ordered = sorted(scenarios, key=lambda sid: sha256(f"{salt}:{sid}".encode()).digest())
    dev: list[int] = []
    train: list[int] = []
    for scenario_id in ordered:
        group = scenarios[scenario_id]
        if len(dev) < dev_targets and len(dev) + len(group) <= dev_targets:
            dev.extend(group)
        elif len(train) + len(group) <= train_targets:
            train.extend(group)
        if len(dev) >= dev_targets and len(train) >= train_targets:
            break
    if len(dev) < dev_targets or len(train) < train_targets:
        raise ValueError(f"cache cannot satisfy scenario-isolated budgets: train={len(train)}/{train_targets}, dev={len(dev)}/{dev_targets}")
    return np.asarray(train, dtype=np.int64), np.asarray(dev, dtype=np.int64)


def local_to_global(points: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) @ rotation_matrix(float(heading)).T + np.asarray(origin, dtype=np.float32)
