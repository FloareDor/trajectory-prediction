from dataclasses import replace

import numpy as np
import torch

from motion_prediction.ablation_data import (
    CacheSpec,
    ShardedCacheWriter,
    ShardedWaymoDataset,
    deterministic_partitions,
    finite_difference_velocity,
    local_to_global,
    scenario_target_record,
    wrap_angle,
)
from motion_prediction.ablation_metrics import paired_bootstrap_ci, per_sample_metrics, summarize_predictions
from motion_prediction.ablation_models import (
    VectorTransformer,
    build_ablation_model,
    masked_smooth_l1,
    multimodal_loss,
    parameter_count,
)
from motion_prediction.ablation_runner import aggregate_results, train_one
from motion_prediction.data import AgentTrack, make_synthetic_scenario
from motion_prediction.waymo_eval import prepare_official_inputs


def _record(seed=0, partial=False):
    scenario = make_synthetic_scenario(seed, history=11, future=80)
    if partial:
        target = scenario.agents[0]
        valid = target.valid.copy()
        valid[-3:] = False
        target = replace(target, valid=valid)
        scenario = replace(scenario, agents=(target,) + scenario.agents[1:])
    return scenario_target_record(scenario, 0)


def _batch(count=2):
    records = [_record(i) for i in range(count)]
    keys = ("target_history", "history_valid", "neighbors", "neighbor_valid", "neighbor_types",
            "target_type", "map_polylines", "map_valid", "map_types", "future", "future_valid")
    return {key: torch.from_numpy(np.stack([record[key] for record in records])) for key in keys}


def test_v2_record_alignment_features_and_partial_future():
    record = _record(1, partial=True)
    assert record["target_history"].shape == (11, 8)
    assert np.allclose(record["target_history"][-1, :2], 0)
    assert record["neighbors"].shape == (8, 11, 8)
    assert record["map_polylines"].shape == (32, 32, 4)
    assert record["future_valid"].sum() == 77
    assert np.all(record["future"][~record["future_valid"]] == 0)
    distances = np.linalg.norm(record["map_polylines"][:, :, :2], axis=-1)
    nearest = [row[mask].min() for row, mask in zip(distances, record["map_valid"]) if mask.any()]
    assert nearest == sorted(nearest)


def test_coordinate_inverse_velocity_and_heading_wrap():
    points = np.array([[1.0, 2.0], [-3.0, 1.0]], np.float32)
    origin = np.array([50.0, -20.0], np.float32)
    heading = 0.7
    local = (points - origin) @ np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]], np.float32).T
    assert np.allclose(local_to_global(local, origin, heading), points, atol=1e-5)
    positions = np.array([[0, 0], [1, 0], [3, 0]], np.float32)
    assert np.allclose(finite_difference_velocity(positions, np.ones(3, bool), 0.5)[:, 0], [2, 2, 4])
    assert np.all(np.abs(wrap_angle(np.array([4 * np.pi, -3 * np.pi]))) <= np.pi)


def test_shards_and_scenario_partitions_are_repeatable(tmp_path):
    writer = ShardedCacheWriter(tmp_path, "train", shard_size=3)
    for seed in range(8):
        writer.add(_record(seed))
    manifest = writer.close()
    dataset = ShardedWaymoDataset(manifest)
    first = deterministic_partitions(dataset, 4, 2, "fixed")
    second = deterministic_partitions(dataset, 4, 2, "fixed")
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    train = {dataset._record(int(i))["scenario_id"].item() for i in first[0]}
    dev = {dataset._record(int(i))["scenario_id"].item() for i in first[1]}
    assert train.isdisjoint(dev)


def test_transformer_shapes_counts_and_context_invariance():
    batch = _batch()
    counts = [parameter_count(build_ablation_model(name)) for name in ("A3", "A4", "A5", "A6")]
    assert len(set(counts)) == 1
    model = VectorTransformer(dropout=0.0, use_neighbors=False, use_map=False, num_modes=1).eval()
    first, _ = model(batch)
    changed = dict(batch)
    changed["neighbors"] = torch.randn_like(batch["neighbors"]) * 100
    changed["map_polylines"] = torch.randn_like(batch["map_polylines"]) * 100
    second, _ = model(changed)
    assert first.shape == (2, 1, 80, 2)
    assert torch.allclose(first, second)
    six, logits = model(batch, num_modes=6)
    assert six.shape == (2, 6, 80, 2)
    assert logits.shape == (2, 6)


def test_masked_losses_and_metrics():
    target = torch.zeros(2, 3, 2)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    prediction = target.clone()
    prediction[~valid] = 100
    assert masked_smooth_l1(prediction, target, valid).item() == 0
    modes = torch.stack([prediction, prediction + 2], dim=1)
    loss, winner = multimodal_loss(modes, torch.zeros(2, 2), target, valid)
    assert winner.tolist() == [0, 0]
    assert loss.item() > 0

    trajectories = modes.numpy()
    summary, per_sample = summarize_predictions(trajectories, np.zeros((2, 2)), target.numpy(), valid.numpy(), np.array([1, 2]))
    assert summary["overall"]["minade"] == 0
    assert summary["primary_macro_minade"] == 0
    assert np.all(per_sample["minfde"] == 0)
    assert np.all(per_sample["brier_minfde"] == 0.25)
    interval = paired_bootstrap_ci(np.ones(20), np.zeros(20), samples=100)
    assert interval["lower"] == interval["upper"] == -1


def test_brier_minfde_uses_endpoint_probability_and_last_valid_point():
    truth = np.zeros((1, 3, 2), np.float32)
    valid = np.array([[True, False, True]])
    perfect = truth[:, None].copy()
    one_mode = per_sample_metrics(perfect, np.array([[100.0]]), truth, valid)
    assert one_mode["minfde"][0] == one_mode["brier_minfde"][0] == 0

    single = perfect + np.array([[[[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]]]], np.float32)
    one_mode = per_sample_metrics(single, np.array([[0.0]]), truth, valid)
    assert one_mode["minfde"][0] == one_mode["brier_minfde"][0] == 3.0

    modes = np.concatenate([perfect, perfect + np.array([3.0, 0.0], np.float32)], axis=1)
    metrics = per_sample_metrics(modes, np.array([[0.0, np.log(9.0)]], np.float32), truth, valid)
    assert np.isclose(metrics["minfde"][0], 0.0)
    assert np.isclose(metrics["brier_minfde"][0], 0.81)


def test_official_input_shape_and_perfect_global_prediction():
    record = _record(4)
    arrays = {
        "trajectories": record["future"][None, None],
        "logits": np.zeros((1, 1), np.float32),
        "origin": record["origin"][None],
        "heading": record["heading"][None],
        "ground_truth": record["ground_truth"][None],
        "ground_truth_valid": record["ground_truth_valid"][None],
        "target_type": record["target_type"][None],
        "object_id": record["object_id"][None],
        "scenario_id": record["scenario_id"][None],
    }
    official = prepare_official_inputs(arrays)
    assert official["prediction_trajectory"].shape == (1, 1, 1, 1, 16, 2)
    truth = official["ground_truth_trajectory"][0, 0, 15::5, :2]
    assert np.allclose(official["prediction_trajectory"][0, 0, 0, 0], truth)


def test_two_epoch_cache_train_predict_smoke(tmp_path):
    writer = ShardedCacheWriter(tmp_path / "cache", "train", shard_size=4)
    for seed in range(8):
        writer.add(_record(seed))
    manifest = writer.close()
    train = ShardedWaymoDataset(manifest, range(6))
    dev = ShardedWaymoDataset(manifest, range(6, 8))
    config = {
        "training": {
            "batch_size": 2, "gradient_accumulation": 1, "learning_rate": 3e-4,
            "weight_decay": 1e-4, "epochs": 2, "warmup_fraction": 0.05,
            "early_stopping_patience": 2, "gradient_clip": 1.0,
            "mixed_precision": False, "num_workers": 0,
        },
        "cache": {"train_manifest": str(manifest), "validation_manifest": str(manifest)},
    }
    run_dir = tmp_path / "run"
    metrics = train_one("A1", 17, train, dev, run_dir, config)
    assert metrics["best_epoch"] in (1, 2)
    assert (run_dir / "checkpoint.pt").is_file()
    assert (run_dir / "predictions.npz").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "curves.json").is_file()


def test_aggregate_report_has_registered_comparisons(tmp_path):
    base_metrics = {
        "primary_macro_minade_8s": 1.0,
        "overall": {"top1_ade": 1.0, "top1_fde": 1.0, "minade": 1.0, "minfde": 1.0, "miss_rate_2m": 0.0},
        "per_type": {name: {"minade": 1.0} for name in ("vehicle", "pedestrian", "cyclist")},
        "horizons": {name: {"minade": 1.0} for name in ("3s", "5s", "8s")},
        "slices": {name: {"minade": 1.0} for name in ("turning", "straight", "interactive", "intersection")},
        "parameter_count": 1,
    }
    for index in range(8):
        directory = tmp_path / f"A{index}" if index == 0 else tmp_path / f"A{index}" / "seed-17"
        directory.mkdir(parents=True)
        metrics = dict(base_metrics)
        metrics["primary_macro_minade_8s"] = 1.0 - index * 0.01
        (directory / "metrics.json").write_text(__import__("json").dumps(metrics), encoding="utf-8")
        with (directory / "predictions.npz").open("wb") as handle:
            np.savez_compressed(
                handle, cache_index=np.arange(3), metric_minade=np.full(3, 1.0 - index * 0.01),
                target_type=np.array([1, 2, 3]), trajectories=np.zeros((3, 1, 8, 2), np.float32),
                logits=np.zeros((3, 1), np.float32), future=np.zeros((3, 8, 2), np.float32),
                future_valid=np.ones((3, 8), bool), heading_change=np.zeros(3), interactive=np.zeros(3, bool), intersection=np.zeros(3, bool),
            )
    report = aggregate_results(tmp_path)
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "combined_context" in text
    assert all(name in text for name in ("minFDE", "Brier-minFDE", "Miss rate", "First-ranked"))
    aggregate = __import__("json").loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert "Brier-minFDE" in aggregate["summary"]["A7"]
    with np.load(tmp_path / "A7" / "seed-17" / "predictions.npz", allow_pickle=False) as data:
        assert "metric_brier_minfde" in data.files
    backfilled = __import__("json").loads((tmp_path / "A7" / "seed-17" / "metrics.json").read_text(encoding="utf-8"))
    assert backfilled["overall"]["brier_minfde"] == 0.0
