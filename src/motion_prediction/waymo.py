"""Optional Waymo TFRecord adapter.

The core package deliberately does not import TensorFlow at module import time,
so synthetic development remains lightweight. Install the ``waymo`` extra to
use this adapter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import AgentTrack, MapPolyline, Scenario, TrafficSignalState, TrajectorySample, scenario_to_sample


def _orient_from_lane_start(points: np.ndarray, lane_points: np.ndarray) -> np.ndarray:
    """Orient a boundary consistently with its lane centerline."""
    if np.linalg.norm(points[0] - lane_points[0]) > np.linalg.norm(points[-1] - lane_points[0]):
        return points[::-1]
    return points


def _road_line_label(line_type: int) -> str:
    names = {
        1: "road_line_broken_single_white",
        2: "road_line_solid_single_white",
        3: "road_line_solid_double_white",
        4: "road_line_broken_single_yellow",
        5: "road_line_broken_double_yellow",
        6: "road_line_solid_single_yellow",
        7: "road_line_solid_double_yellow",
        8: "road_line_passing_double_yellow",
    }
    return names.get(line_type, "road_line")


def load_tfrecord_scenarios(path: str | Path, limit: int | None = None) -> list[Scenario]:
    """Read Scenario protobufs from a Waymo Motion TFRecord shard.

    Waymo package versions have changed protobuf import paths, so imports are
    kept inside this function and a useful install message is raised when the
    optional dependencies are missing.
    """
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as exc:  # pragma: no cover - requires optional package
        raise RuntimeError("Install the optional Waymo dependencies with `pip install -e .[waymo]`.") from exc

    scenarios: list[Scenario] = []
    for raw in tf.data.TFRecordDataset([str(path)]):
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(bytes(raw.numpy()))
        agents: list[AgentTrack] = []
        for track_index, track in enumerate(proto.tracks):
            positions = []
            valid = []
            headings = []
            velocities = []
            lengths = []
            widths = []
            for state in track.states:
                positions.append((state.center_x, state.center_y))
                valid.append(bool(state.valid))
                headings.append(state.heading)
                velocities.append((state.velocity_x, state.velocity_y))
                lengths.append(state.length)
                widths.append(state.width)
            last_state = track.states[-1]
            agents.append(
                AgentTrack(
                    track.id,
                    str(track.object_type),
                    np.asarray(positions, np.float32),
                    np.asarray(valid, bool),
                    np.asarray(headings, np.float32),
                    length=float(last_state.length) if last_state.length else 4.5,
                    width=float(last_state.width) if last_state.width else 1.8,
                    is_ego=track_index == proto.sdc_track_index,
                    velocities=np.asarray(velocities, np.float32),
                    lengths=np.asarray(lengths, np.float32),
                    widths=np.asarray(widths, np.float32),
                )
            )
        map_polylines: list[MapPolyline] = []
        boundary_features: dict[int, np.ndarray] = {}
        lane_features = []
        controlled_lane_ids = {
            lane_state.lane
            for dynamic_state in proto.dynamic_map_states
            for lane_state in dynamic_state.lane_states
        }
        for feature in proto.map_features:
            # The Waymo map proto stores most vector features as repeated
            # points. Keep this conversion intentionally small and typed; the
            # model only needs local XY geometry at this stage.
            for field_name, label in (
                ("lane", "lane_center"),
                ("road_line", "road_line"),
                ("road_edge", "road_edge"),
                ("crosswalk", "crosswalk"),
                ("speed_bump", "speed_bump"),
            ):
                if not feature.HasField(field_name):
                    continue
                value = getattr(feature, field_name)
                points = getattr(value, "polyline", None) or getattr(value, "polygon", None)
                if points:
                    array = np.asarray([(point.x, point.y) for point in points], np.float32)
                    feature_label = _road_line_label(int(value.type)) if field_name == "road_line" else label
                    if field_name == "lane" and feature.id in controlled_lane_ids:
                        feature_label = "controlled_lane"
                    map_polylines.append(MapPolyline(array, feature_label))
                    if field_name in {"road_line", "road_edge"}:
                        boundary_features[feature.id] = array
                    if field_name == "lane":
                        lane_features.append((feature.lane, array))
            if feature.HasField("stop_sign"):
                position = feature.stop_sign.position
                map_polylines.append(MapPolyline(np.asarray([[position.x, position.y]], np.float32), "stop_sign"))
        road_polygons: list[np.ndarray] = []
        for lane, lane_points in lane_features:
            # BoundarySegment points to RoadLine/RoadEdge feature IDs. Using
            # their real geometry gives us lane corridors rather than an
            # invented rasterized road surface.
            left_ids = [segment.boundary_feature_id for segment in lane.left_boundaries if segment.boundary_feature_id in boundary_features]
            right_ids = [segment.boundary_feature_id for segment in lane.right_boundaries if segment.boundary_feature_id in boundary_features]
            if not left_ids or not right_ids:
                continue
            left = _orient_from_lane_start(boundary_features[left_ids[0]], lane_points)
            right = _orient_from_lane_start(boundary_features[right_ids[0]], lane_points)
            if len(left) > 1 and len(right) > 1:
                road_polygons.append(np.concatenate([left, right[::-1]], axis=0))
        traffic_signal_states: list[tuple[TrafficSignalState, ...]] = []
        for dynamic_state in proto.dynamic_map_states:
            signals: list[TrafficSignalState] = []
            for lane_state in dynamic_state.lane_states:
                signals.append(
                    TrafficSignalState(
                        lane_id=lane_state.lane,
                        state=int(lane_state.state),
                        stop_point=np.asarray([lane_state.stop_point.x, lane_state.stop_point.y], np.float32),
                    )
                )
            traffic_signal_states.append(tuple(signals))
        prediction_target_ids = tuple(
            agents[required.track_index].object_id
            for required in proto.tracks_to_predict
            if 0 <= required.track_index < len(agents)
        )
        scenarios.append(
            Scenario(
                scenario_id=proto.scenario_id,
                agents=tuple(agents),
                map_polylines=tuple(map_polylines),
                traffic_signal_states=tuple(traffic_signal_states),
                road_polygons=tuple(road_polygons),
                current_time_index=int(proto.current_time_index),
                prediction_target_ids=prediction_target_ids,
            )
        )
        if limit is not None and len(scenarios) >= limit:
            break
    return scenarios


def preprocess_tfrecords(
    paths: list[str | Path],
    history_steps: int = 11,
    future_steps: int = 80,
    max_scenarios: int | None = None,
) -> tuple[list[TrajectorySample], dict[str, int]]:
    """Convert Waymo prediction targets into fixed-shape training samples.

    Phase one keeps only targets that are valid for the complete requested
    window. Future masking can be introduced later without silently training
    against invalid coordinates now.
    """

    samples: list[TrajectorySample] = []
    stats = {"files": 0, "scenarios": 0, "target_candidates": 0, "samples": 0, "skipped_invalid": 0}
    for path in paths:
        if max_scenarios is not None and stats["scenarios"] >= max_scenarios:
            break
        remaining = None if max_scenarios is None else max_scenarios - stats["scenarios"]
        scenarios = load_tfrecord_scenarios(path, limit=remaining)
        stats["files"] += 1
        for scenario in scenarios:
            stats["scenarios"] += 1
            stats["target_candidates"] += len(scenario.prediction_target_ids)
            for target_id in scenario.prediction_target_ids:
                try:
                    samples.append(scenario_to_sample(scenario, target_id, history_steps, future_steps))
                except ValueError:
                    stats["skipped_invalid"] += 1
    stats["samples"] = len(samples)
    return samples, stats


def preprocess_waymo_cache(
    paths: list[str | Path],
    output_dir: str | Path,
    split: str,
    history_steps: int = 11,
    future_steps: int = 80,
    shard_size: int = 1024,
    max_scenarios: int | None = None,
) -> tuple[Path, dict[str, int]]:
    """Create a version-2 sharded cache for every official prediction target."""
    from .ablation_data import CacheSpec, ShardedCacheWriter, scenario_target_record

    spec = CacheSpec(history_steps=history_steps, future_steps=future_steps)
    writer = ShardedCacheWriter(output_dir, split=split, spec=spec, shard_size=shard_size)
    stats = {
        "files": 0,
        "scenarios": 0,
        "target_candidates": 0,
        "samples": 0,
        "skipped_invalid_current": 0,
        "partial_histories": 0,
        "partial_futures": 0,
    }
    for path in paths:
        if max_scenarios is not None and stats["scenarios"] >= max_scenarios:
            break
        remaining = None if max_scenarios is None else max_scenarios - stats["scenarios"]
        scenarios = load_tfrecord_scenarios(path, limit=remaining)
        stats["files"] += 1
        for scenario in scenarios:
            stats["scenarios"] += 1
            stats["target_candidates"] += len(scenario.prediction_target_ids)
            for target_id in scenario.prediction_target_ids:
                try:
                    record = scenario_target_record(scenario, target_id, spec)
                except ValueError:
                    stats["skipped_invalid_current"] += 1
                    continue
                stats["partial_histories"] += int(not record["history_valid"].all())
                stats["partial_futures"] += int(not record["future_valid"].all())
                writer.add(record)
                stats["samples"] += 1
    if not stats["samples"]:
        raise ValueError(f"no valid current-frame targets found; stats={stats}")
    return writer.close(stats), stats
