"""download and preprocess Argoverse 2 motion forecasting data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import shutil
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq

from .ablation_data import CacheSpec, ShardedCacheWriter, scenario_target_record
from .data import AgentTrack, MapPolyline, Scenario


BUCKET = "https://argoverse.s3.amazonaws.com"
PREFIX = "datasets/av2/motion-forecasting"


def list_av2_scenarios(split: str, limit: int | None = None) -> list[str]:
    split = "val" if split == "validation" else split
    prefix = f"{PREFIX}/{split}/"
    names: list[str] = []
    token = None
    while limit is None or len(names) < limit:
        query = {"list-type": 2, "prefix": prefix, "delimiter": "/", "max-keys": 1000}
        if token:
            query["continuation-token"] = token
        with urlopen(f"{BUCKET}/?{urlencode(query)}") as response:
            root = ET.parse(response).getroot()
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for item in root.findall("s3:CommonPrefixes/s3:Prefix", namespace):
            names.append(item.text.rstrip("/").split("/")[-1])
            if limit is not None and len(names) >= limit:
                break
        next_token = root.findtext("s3:NextContinuationToken", namespaces=namespace)
        if not next_token:
            break
        token = next_token
    return names


def _download(url: str, path: Path) -> None:
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(path)


def download_av2(split: str, output_dir: str | Path, max_scenarios: int, workers: int = 16) -> Path:
    remote_split = "val" if split == "validation" else split
    root = Path(output_dir) / remote_split
    scenario_ids = list_av2_scenarios(remote_split, max_scenarios)

    def download_scenario(scenario_id: str) -> None:
        folder = root / scenario_id
        base = f"{BUCKET}/{PREFIX}/{remote_split}/{scenario_id}"
        _download(f"{base}/scenario_{scenario_id}.parquet", folder / f"scenario_{scenario_id}.parquet")
        _download(f"{base}/log_map_archive_{scenario_id}.json", folder / f"log_map_archive_{scenario_id}.json")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, _ in enumerate(pool.map(download_scenario, scenario_ids), 1):
            if count % 1000 == 0 or count == len(scenario_ids):
                print(f"downloaded {count}/{len(scenario_ids)} scenarios", flush=True)
    manifest = root / "download_manifest.json"
    manifest.write_text(json.dumps({"dataset": "av2", "split": remote_split, "count": len(scenario_ids), "scenarios": scenario_ids}, indent=2), encoding="utf-8")
    return manifest


def _object_id(track_id: str) -> int:
    try:
        return int(track_id)
    except ValueError:
        return int.from_bytes(sha256(track_id.encode()).digest()[:8], "little") & (2**63 - 1)


def _agent_size(object_type: str) -> tuple[float, float]:
    if object_type in {"pedestrian"}:
        return 0.7, 0.7
    if object_type in {"cyclist", "motorcyclist", "riderless_bicycle"}:
        return 2.0, 0.7
    if object_type == "bus":
        return 12.0, 2.6
    return 4.5, 1.8


def _agent_type(object_type: str) -> str:
    if object_type == "pedestrian":
        return "pedestrian"
    if object_type in {"cyclist", "motorcyclist", "riderless_bicycle"}:
        return "cyclist"
    if object_type in {"vehicle", "bus"}:
        return "vehicle"
    return "other"


def _points(values: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([[point["x"], point["y"]] for point in values], dtype=np.float32)


def _line_type(mark: str) -> str:
    names = {
        "DASHED_WHITE": "road_line_broken_single_white",
        "SOLID_WHITE": "road_line_solid_single_white",
        "DOUBLE_SOLID_WHITE": "road_line_solid_double_white",
        "DASHED_YELLOW": "road_line_broken_single_yellow",
        "SOLID_YELLOW": "road_line_solid_single_yellow",
        "DOUBLE_SOLID_YELLOW": "road_line_solid_double_yellow",
    }
    return names.get(mark, "road_edge")


def load_av2_scenario(path: str | Path) -> Scenario:
    parquet_path = Path(path)
    values = pq.read_table(parquet_path).to_pydict()
    total_steps = int(values["num_timestamps"][0])
    scenario_id = str(values["scenario_id"][0])
    focal_track_id = str(values["focal_track_id"][0])
    rows_by_track: dict[str, list[int]] = {}
    for row, track_id in enumerate(values["track_id"]):
        rows_by_track.setdefault(str(track_id), []).append(row)

    agents = []
    focal_id = _object_id(focal_track_id)
    for track_id, rows in rows_by_track.items():
        positions = np.zeros((total_steps, 2), np.float32)
        velocities = np.zeros((total_steps, 2), np.float32)
        headings = np.zeros(total_steps, np.float32)
        valid = np.zeros(total_steps, bool)
        timesteps = np.asarray([values["timestep"][row] for row in rows], dtype=int)
        positions[timesteps, 0] = [values["position_x"][row] for row in rows]
        positions[timesteps, 1] = [values["position_y"][row] for row in rows]
        velocities[timesteps, 0] = [values["velocity_x"][row] for row in rows]
        velocities[timesteps, 1] = [values["velocity_y"][row] for row in rows]
        headings[timesteps] = [values["heading"][row] for row in rows]
        valid[timesteps] = True
        raw_type = str(values["object_type"][rows[0]]).lower()
        length, width = _agent_size(raw_type)
        agents.append(AgentTrack(
            _object_id(track_id), _agent_type(raw_type), positions, valid, headings,
            length, width, False, velocities,
        ))

    map_path = parquet_path.with_name(f"log_map_archive_{scenario_id}.json")
    map_data = json.loads(map_path.read_text(encoding="utf-8"))
    polylines: list[MapPolyline] = []
    for lane in map_data["lane_segments"].values():
        lane_type = "controlled_lane" if lane["is_intersection"] else "lane_center"
        polylines.append(MapPolyline(_points(lane["centerline"]), lane_type))
        for side in ("left", "right"):
            boundary = lane[f"{side}_lane_boundary"]
            if boundary:
                polylines.append(MapPolyline(_points(boundary), _line_type(lane[f"{side}_lane_mark_type"])))
    for crossing in map_data["pedestrian_crossings"].values():
        polygon = np.concatenate([_points(crossing["edge1"]), _points(crossing["edge2"])[::-1]])
        polylines.append(MapPolyline(polygon, "crosswalk"))
    return Scenario(
        scenario_id=scenario_id,
        agents=tuple(agents),
        map_polylines=tuple(polylines),
        timestep_seconds=0.1,
        current_time_index=49,
        prediction_target_ids=(focal_id,),
    )


def preprocess_av2_cache(
    input_dir: str | Path,
    output_dir: str | Path,
    split: str,
    shard_size: int = 1024,
    max_scenarios: int | None = None,
) -> tuple[Path, dict[str, int]]:
    paths = sorted(Path(input_dir).rglob("scenario_*.parquet"))
    if max_scenarios is not None:
        paths = paths[:max_scenarios]
    if not paths:
        raise FileNotFoundError(f"no AV2 parquet files found under {input_dir}")
    spec = CacheSpec(history_steps=50, future_steps=60)
    writer = ShardedCacheWriter(output_dir, split, spec, shard_size)
    type_counts = {"vehicle": 0, "pedestrian": 0, "cyclist": 0, "other": 0}
    for count, path in enumerate(paths, 1):
        scenario = load_av2_scenario(path)
        target_id = scenario.prediction_target_ids[0]
        target = next(agent for agent in scenario.agents if agent.object_id == target_id)
        writer.add(scenario_target_record(scenario, target_id, spec))
        type_counts[target.object_type] += 1
        if count % 1000 == 0 or count == len(paths):
            print(f"processed {count}/{len(paths)} scenarios", flush=True)
    stats = {"scenarios": len(paths), "targets": len(paths), **type_counts}
    return writer.close(stats), stats
