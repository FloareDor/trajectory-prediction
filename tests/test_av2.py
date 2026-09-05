import json

import pyarrow as pa
import pyarrow.parquet as pq

from motion_prediction.ablation_data import CacheSpec, scenario_target_record
from motion_prediction.av2 import load_av2_scenario


def test_av2_parquet_and_map_are_loaded(tmp_path):
    scenario_id = "small-scene"
    folder = tmp_path / scenario_id
    folder.mkdir()
    steps = list(range(110))
    table = pa.table({
        "track_id": ["target"] * 110,
        "object_type": ["pedestrian"] * 110,
        "timestep": steps,
        "position_x": [step * 0.1 for step in steps],
        "position_y": [0.0] * 110,
        "heading": [0.0] * 110,
        "velocity_x": [1.0] * 110,
        "velocity_y": [0.0] * 110,
        "scenario_id": [scenario_id] * 110,
        "num_timestamps": [110] * 110,
        "focal_track_id": ["target"] * 110,
    })
    parquet = folder / f"scenario_{scenario_id}.parquet"
    pq.write_table(table, parquet)
    map_data = {
        "lane_segments": {"1": {
            "centerline": [{"x": 0, "y": 0}, {"x": 10, "y": 0}],
            "is_intersection": True,
            "left_lane_boundary": [{"x": 0, "y": 1}, {"x": 10, "y": 1}],
            "right_lane_boundary": [],
            "left_lane_mark_type": "DASHED_WHITE",
            "right_lane_mark_type": "NONE",
        }},
        "pedestrian_crossings": {},
    }
    (folder / f"log_map_archive_{scenario_id}.json").write_text(json.dumps(map_data), encoding="utf-8")
    scenario = load_av2_scenario(parquet)
    record = scenario_target_record(scenario, scenario.prediction_target_ids[0], CacheSpec(history_steps=50, future_steps=60))
    assert record["target_history"].shape == (50, 8)
    assert record["future"].shape == (60, 2)
    assert record["future_valid"].all()
    assert record["target_type"] == 2
    assert record["intersection"]
