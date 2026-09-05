"""Dataset-neutral scene structures and a deterministic synthetic dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class AgentTrack:
    object_id: int
    object_type: str
    positions: np.ndarray  # [time, 2], global XY coordinates
    valid: np.ndarray  # [time]
    heading: np.ndarray | None = None  # [time], radians
    length: float = 4.5  # metres; used for bird's-eye-view actor footprints
    width: float = 1.8  # metres
    is_ego: bool = False
    velocities: np.ndarray | None = None  # [time, 2], global metres/second
    lengths: np.ndarray | None = None  # [time]
    widths: np.ndarray | None = None  # [time]


@dataclass(frozen=True)
class MapPolyline:
    points: np.ndarray  # [points, 2]
    polyline_type: str = "lane_center"


@dataclass(frozen=True)
class TrafficSignalState:
    """State of a signal controlling a lane at a particular timestep."""

    lane_id: int
    state: int  # Waymo TrafficSignalLaneState.State enum value
    stop_point: np.ndarray  # [2]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    agents: tuple[AgentTrack, ...]
    map_polylines: tuple[MapPolyline, ...] = ()
    traffic_signal_states: tuple[tuple[TrafficSignalState, ...], ...] = ()
    road_polygons: tuple[np.ndarray, ...] = ()  # lane corridors derived from actual map boundaries
    timestep_seconds: float = 0.1
    current_time_index: int | None = None
    prediction_target_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrajectorySample:
    scenario_id: str
    target_id: int
    history: np.ndarray  # [history, 2], target-centric
    future: np.ndarray  # [future, 2], target-centric
    neighbors: np.ndarray  # [neighbors, history, 2]
    neighbor_mask: np.ndarray  # [neighbors, history]
    map_polylines: np.ndarray  # [polylines, points, 2]
    map_mask: np.ndarray  # [polylines, points]
    map_types: tuple[str, ...]  # [polylines]
    target_heading: float  # radians in the target-centric frame
    target_length: float
    target_width: float
    target_is_ego: bool
    neighbor_headings: np.ndarray  # [neighbors], radians in target-centric frame
    neighbor_lengths: np.ndarray  # [neighbors]
    neighbor_widths: np.ndarray  # [neighbors]
    neighbor_is_ego: np.ndarray  # [neighbors]
    traffic_light_positions: np.ndarray  # [traffic_lights, 2]
    traffic_light_states: np.ndarray  # [traffic_lights], Waymo enum values
    road_polygons: np.ndarray  # [road_polygons, points, 2]
    road_polygon_mask: np.ndarray  # [road_polygons, points]


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ rotation.T


def make_synthetic_scenario(
    seed: int = 0,
    num_agents: int = 4,
    history: int = 10,
    future: int = 20,
) -> Scenario:
    """Create a small scene with straight and gently turning trajectories.

    This fixture deliberately has the same shape as a real scene, making it
    useful for local development and end-to-end tests without dataset access.
    """

    rng = np.random.default_rng(seed)
    total = history + future
    agents: list[AgentTrack] = []
    for object_id in range(num_agents):
        start = rng.normal(0.0, 10.0, size=2).astype(np.float32)
        velocity = rng.uniform(2.0, 8.0) * np.array(
            [np.cos(rng.uniform(-0.4, 0.4)), np.sin(rng.uniform(-0.4, 0.4))],
            dtype=np.float32,
        )
        # A visible but learnable turn: its effect is already present in the
        # observed history, while constant velocity cannot extrapolate it.
        curvature = rng.normal(0.0, 0.03)
        t = np.arange(total, dtype=np.float32)
        x = start[0] + velocity[0] * t + curvature * t * t
        y = start[1] + velocity[1] * t + 0.5 * curvature * t * t
        positions = np.stack([x, y], axis=-1).astype(np.float32)
        heading = np.full(total, np.arctan2(velocity[1], velocity[0]), dtype=np.float32)
        agents.append(AgentTrack(object_id, "vehicle", positions, np.ones(total, bool), heading, is_ego=object_id == 1))

    horizontal = np.linspace(-100, 100, 81, dtype=np.float32)
    vertical = np.linspace(-100, 100, 81, dtype=np.float32)
    polylines = (
        MapPolyline(np.stack([horizontal, np.zeros(81)], axis=-1), "lane_center"),
        MapPolyline(np.stack([np.zeros(81), vertical], axis=-1), "lane_center"),
        MapPolyline(np.stack([horizontal, np.full(81, -5.0)], axis=-1), "road_line_solid_single_white"),
        MapPolyline(np.stack([horizontal, np.full(81, 5.0)], axis=-1), "road_line_solid_single_white"),
        MapPolyline(np.stack([np.full(81, -5.0), vertical], axis=-1), "road_line_solid_single_white"),
        MapPolyline(np.stack([np.full(81, 5.0), vertical], axis=-1), "road_line_solid_single_white"),
    )
    road_polygons = (
        np.array([[-100.0, -5.0], [100.0, -5.0], [100.0, 5.0], [-100.0, 5.0]], dtype=np.float32),
        np.array([[-5.0, -100.0], [5.0, -100.0], [5.0, 100.0], [-5.0, 100.0]], dtype=np.float32),
    )
    traffic_states = tuple(
        (TrafficSignalState(lane_id=1, state=6, stop_point=np.array([35.0, 2.0], dtype=np.float32)),)
        for _ in range(total)
    )
    return Scenario(
        f"synthetic-{seed}",
        tuple(agents),
        polylines,
        traffic_states,
        road_polygons,
        current_time_index=history - 1,
        prediction_target_ids=(0,),
    )


def scenario_to_sample(
    scenario: Scenario,
    target_id: int = 0,
    history_steps: int = 10,
    future_steps: int = 20,
    max_neighbors: int = 8,
    max_polylines: int = 32,
    polyline_points: int = 32,
    max_traffic_lights: int = 16,
    max_road_polygons: int = 32,
    road_polygon_points: int = 64,
) -> TrajectorySample:
    """Convert a global-coordinate scenario into a target-centric sample."""

    target = next((agent for agent in scenario.agents if agent.object_id == target_id), None)
    if target is None:
        raise ValueError(f"target {target_id} is not present in scenario {scenario.scenario_id}")
    current_index = scenario.current_time_index if scenario.current_time_index is not None else history_steps - 1
    history_start = current_index - history_steps + 1
    future_end = current_index + future_steps + 1
    if history_start < 0 or future_end > target.positions.shape[0]:
        raise ValueError(
            f"target has {target.positions.shape[0]} steps, but history [{history_start}, {current_index}] "
            f"and future [{current_index + 1}, {future_end - 1}] were requested"
        )
    target_window = target.valid[history_start:future_end]
    if len(target_window) != history_steps + future_steps or not target_window.all():
        raise ValueError(f"target {target_id} does not have a fully valid requested trajectory")
    origin = target.positions[current_index].astype(np.float32)
    heading = float(target.heading[current_index]) if target.heading is not None else 0.0
    # Rotate by the negative heading so +x points along the target's motion.
    transform = lambda pts: _rotate(np.asarray(pts, dtype=np.float32) - origin, -heading)
    history = transform(target.positions[history_start : current_index + 1])
    future = transform(target.positions[current_index + 1 : future_end])

    neighbors = np.zeros((max_neighbors, history_steps, 2), dtype=np.float32)
    neighbor_mask = np.zeros((max_neighbors, history_steps), dtype=bool)
    neighbor_headings = np.zeros(max_neighbors, dtype=np.float32)
    neighbor_lengths = np.full(max_neighbors, 4.5, dtype=np.float32)
    neighbor_widths = np.full(max_neighbors, 1.8, dtype=np.float32)
    neighbor_is_ego = np.zeros(max_neighbors, dtype=bool)
    other_agents = [
        agent
        for agent in scenario.agents
        if agent.object_id != target_id and current_index < len(agent.positions) and agent.valid[current_index]
    ]
    other_agents.sort(key=lambda a: float(np.linalg.norm(a.positions[current_index] - origin)))
    for index, agent in enumerate(other_agents[:max_neighbors]):
        agent_start = max(0, history_start)
        agent_end = min(current_index + 1, agent.positions.shape[0])
        count = max(0, agent_end - agent_start)
        destination_start = history_steps - count
        neighbors[index, destination_start:] = transform(agent.positions[agent_start:agent_end])
        neighbor_mask[index, destination_start:] = agent.valid[agent_start:agent_end]
        neighbor_headings[index] = (float(agent.heading[current_index]) - heading) if agent.heading is not None else 0.0
        neighbor_lengths[index] = agent.length
        neighbor_widths[index] = agent.width
        neighbor_is_ego[index] = agent.is_ego

    map_array = np.zeros((max_polylines, polyline_points, 2), dtype=np.float32)
    map_mask = np.zeros((max_polylines, polyline_points), dtype=bool)
    map_types = ["" for _ in range(max_polylines)]
    nearby_polylines = sorted(
        scenario.map_polylines,
        key=lambda polyline: float(np.linalg.norm(polyline.points - origin, axis=-1).min()),
    )
    for poly_index, polyline in enumerate(nearby_polylines[:max_polylines]):
        points = polyline.points
        indices = np.linspace(0, len(points) - 1, min(polyline_points, len(points))).round().astype(int)
        count = len(indices)
        map_array[poly_index, :count] = transform(points[indices])
        map_mask[poly_index, :count] = True
        map_types[poly_index] = polyline.polyline_type
    traffic_light_positions = np.zeros((max_traffic_lights, 2), dtype=np.float32)
    traffic_light_states = np.zeros(max_traffic_lights, dtype=np.int64)
    if scenario.traffic_signal_states:
        current_states = scenario.traffic_signal_states[min(current_index, len(scenario.traffic_signal_states) - 1)]
        for index, signal in enumerate(current_states[:max_traffic_lights]):
            traffic_light_positions[index] = transform(signal.stop_point)
            traffic_light_states[index] = signal.state
    road_polygon_array = np.zeros((max_road_polygons, road_polygon_points, 2), dtype=np.float32)
    road_polygon_mask = np.zeros((max_road_polygons, road_polygon_points), dtype=bool)
    for index, polygon in enumerate(scenario.road_polygons[:max_road_polygons]):
        indices = np.linspace(0, len(polygon) - 1, min(road_polygon_points, len(polygon))).round().astype(int)
        count = len(indices)
        road_polygon_array[index, :count] = transform(polygon[indices])
        road_polygon_mask[index, :count] = True
    return TrajectorySample(
        scenario.scenario_id,
        target_id,
        history,
        future,
        neighbors,
        neighbor_mask,
        map_array,
        map_mask,
        tuple(map_types),
        0.0,
        target.length,
        target.width,
        target.is_ego,
        neighbor_headings,
        neighbor_lengths,
        neighbor_widths,
        neighbor_is_ego,
        traffic_light_positions,
        traffic_light_states,
        road_polygon_array,
        road_polygon_mask,
    )


def save_trajectory_samples(samples: list[TrajectorySample], path: str | Path) -> None:
    """Write fixed-shape trajectory samples to a compressed, pickle-free cache."""

    if not samples:
        raise ValueError("cannot save an empty trajectory cache")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scalar_fields = (
        "target_id",
        "target_heading",
        "target_length",
        "target_width",
        "target_is_ego",
    )
    array_fields = (
        "history",
        "future",
        "neighbors",
        "neighbor_mask",
        "map_polylines",
        "map_mask",
        "neighbor_headings",
        "neighbor_lengths",
        "neighbor_widths",
        "neighbor_is_ego",
        "traffic_light_positions",
        "traffic_light_states",
        "road_polygons",
        "road_polygon_mask",
    )
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray(1, dtype=np.int64),
        "scenario_id": np.asarray([sample.scenario_id for sample in samples]),
        "map_types": np.asarray([sample.map_types for sample in samples]),
    }
    for field in scalar_fields:
        payload[field] = np.asarray([getattr(sample, field) for sample in samples])
    for field in array_fields:
        payload[field] = np.stack([getattr(sample, field) for sample in samples])
    with output.open("wb") as handle:
        np.savez_compressed(handle, **payload)


def load_trajectory_samples(path: str | Path) -> list[TrajectorySample]:
    """Load samples created by :func:`save_trajectory_samples`."""

    with np.load(Path(path), allow_pickle=False) as payload:
        if int(payload["format_version"]) != 1:
            raise ValueError(f"unsupported trajectory cache version in {path}")
        samples: list[TrajectorySample] = []
        for index in range(len(payload["history"])):
            samples.append(
                TrajectorySample(
                    scenario_id=str(payload["scenario_id"][index]),
                    target_id=int(payload["target_id"][index]),
                    history=payload["history"][index],
                    future=payload["future"][index],
                    neighbors=payload["neighbors"][index],
                    neighbor_mask=payload["neighbor_mask"][index],
                    map_polylines=payload["map_polylines"][index],
                    map_mask=payload["map_mask"][index],
                    map_types=tuple(str(value) for value in payload["map_types"][index]),
                    target_heading=float(payload["target_heading"][index]),
                    target_length=float(payload["target_length"][index]),
                    target_width=float(payload["target_width"][index]),
                    target_is_ego=bool(payload["target_is_ego"][index]),
                    neighbor_headings=payload["neighbor_headings"][index],
                    neighbor_lengths=payload["neighbor_lengths"][index],
                    neighbor_widths=payload["neighbor_widths"][index],
                    neighbor_is_ego=payload["neighbor_is_ego"][index],
                    traffic_light_positions=payload["traffic_light_positions"][index],
                    traffic_light_states=payload["traffic_light_states"][index],
                    road_polygons=payload["road_polygons"][index],
                    road_polygon_mask=payload["road_polygon_mask"][index],
                )
            )
    return samples


class SyntheticDataset:
    """Iterable collection of preprocessed synthetic samples."""

    def __init__(self, count: int = 64, history_steps: int = 10, future_steps: int = 20):
        self.samples = [
            scenario_to_sample(
                make_synthetic_scenario(i, history=history_steps, future=future_steps),
                target_id=0,
                history_steps=history_steps,
                future_steps=future_steps,
            )
            for i in range(count)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrajectorySample:
        return self.samples[index]

    def __iter__(self) -> Iterator[TrajectorySample]:
        return iter(self.samples)
