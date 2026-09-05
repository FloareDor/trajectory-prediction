"""Command-line entry points for a local end-to-end smoke run."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from .baselines import constant_velocity
from .data import SyntheticDataset, save_trajectory_samples
from .metrics import ade, fde
from .train import train_history_model, train_social_map_model
from .visualize import plot_prediction, plot_scene


def _run_training(output: Path, epochs: int) -> None:
    dataset = SyntheticDataset(count=96)
    split = int(len(dataset) * 0.8)
    train_history_model(list(dataset.samples[:split]), list(dataset.samples[split:]), output, epochs=epochs)
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    from .models import HistoryGRU

    model = HistoryGRU(checkpoint["hidden_size"], checkpoint["future_steps"])
    model.load_state_dict(checkpoint["model_state"])
    sample = dataset.samples[-1]
    with torch.no_grad():
        prediction = model(torch.from_numpy(sample.history)[None]).numpy()[0]
    baseline = constant_velocity(sample.history, sample.future.shape[0])
    plot_prediction(sample, prediction, output / "prediction.png", baseline)
    print(f"Wrote {output / 'best_model.pt'}")
    print(f"Wrote {output / 'metrics.json'}")
    print(f"Wrote {output / 'prediction.png'}")


def _run_experiment(output: Path, epochs: int, count: int, history_steps: int, future_steps: int, modes: int) -> None:
    """Run comparable baseline, history, and social/map experiments."""
    dataset = SyntheticDataset(count=count, history_steps=history_steps, future_steps=future_steps)
    split = int(len(dataset) * 0.8)
    train_samples = list(dataset.samples[:split])
    validation_samples = list(dataset.samples[split:])
    future_steps = validation_samples[0].future.shape[0]
    target = np.stack([sample.future for sample in validation_samples])
    baseline = np.stack([constant_velocity(sample.history, future_steps) for sample in validation_samples])
    summary: dict[str, object] = {
        "dataset": "synthetic",
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "epochs": epochs,
        "history_steps": history_steps,
        "future_steps": future_steps,
        "modes": modes,
        "constant_velocity": {"ade": ade(baseline, target), "fde": fde(baseline, target)},
    }
    history_dir = output / "history_gru"
    _, history_metrics = train_history_model(train_samples, validation_samples, history_dir, epochs=epochs)
    social_dir = output / "social_map"
    _, social_metrics = train_social_map_model(train_samples, validation_samples, social_dir, epochs=epochs, modes=modes)
    summary["history_gru"] = history_metrics
    summary["social_map"] = social_metrics
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {output / 'summary.json'}")


def _waymo_sample(scenario_path: Path, scenario_index: int, target_id: int | None, history_steps: int, future_steps: int):
    from .data import scenario_to_sample
    from .waymo import load_tfrecord_scenarios

    scenarios = load_tfrecord_scenarios(scenario_path, limit=scenario_index + 1)
    if scenario_index >= len(scenarios):
        raise ValueError(f"TFRecord only contained {len(scenarios)} scenario(s); requested index {scenario_index}.")
    scenario = scenarios[scenario_index]
    chosen_target = target_id
    if chosen_target is None:
        chosen_target = scenario.prediction_target_ids[0] if scenario.prediction_target_ids else scenario.agents[0].object_id
    return scenario_to_sample(scenario, chosen_target, history_steps, future_steps)


def _expand_waymo_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        if value.is_dir():
            paths.extend(sorted(path for path in value.rglob("*") if path.is_file() and "tfrecord" in path.name.lower()))
        elif value.is_file():
            paths.append(value)
        else:
            paths.extend(Path(match) for match in sorted(glob.glob(str(value))))
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise FileNotFoundError("no Waymo TFRecord files matched --input")
    return unique_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="motion-prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="train, evaluate, and plot on synthetic data")
    smoke.add_argument("--output", type=Path, default=Path("artifacts"))
    smoke.add_argument("--epochs", type=int, default=20)
    train = subparsers.add_parser("train-synthetic", help="train the local synthetic model")
    train.add_argument("--output", type=Path, default=Path("artifacts"))
    train.add_argument("--epochs", type=int, default=20)
    social = subparsers.add_parser("train-social-synthetic", help="train the social/map multimodal model")
    social.add_argument("--output", type=Path, default=Path("artifacts"))
    social.add_argument("--epochs", type=int, default=20)
    social.add_argument("--modes", type=int, default=3)
    experiment = subparsers.add_parser("experiment", help="run comparable baseline/model experiments")
    experiment.add_argument("--output", type=Path, default=Path("experiments/first-run"))
    experiment.add_argument("--epochs", type=int, default=20)
    experiment.add_argument("--count", type=int, default=96)
    experiment.add_argument("--history-steps", type=int, default=10)
    experiment.add_argument("--future-steps", type=int, default=20)
    experiment.add_argument("--modes", type=int, default=3)
    predict = subparsers.add_parser("predict-synthetic", help="render one synthetic prediction from a checkpoint")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("artifacts/prediction.png"))
    predict.add_argument("--seed", type=int, default=95)
    scene = subparsers.add_parser("visualize-synthetic", help="render a scene without running a model")
    scene.add_argument("--output", type=Path, default=Path("artifacts/scene.png"))
    scene.add_argument("--seed", type=int, default=95)
    waymo_scene = subparsers.add_parser("visualize-waymo", help="render a Waymo scene from one TFRecord")
    waymo_scene.add_argument("--scenario", type=Path, required=True)
    waymo_scene.add_argument("--output", type=Path, default=Path("artifacts/waymo_scene.png"))
    waymo_scene.add_argument("--scenario-index", type=int, default=0)
    waymo_scene.add_argument("--target-id", type=int)
    waymo_scene.add_argument("--history-steps", type=int, default=10)
    waymo_scene.add_argument("--future-steps", type=int, default=80)
    waymo_predict = subparsers.add_parser("predict-waymo", help="render history-model predictions for one Waymo scenario")
    waymo_predict.add_argument("--scenario", type=Path, required=True)
    waymo_predict.add_argument("--checkpoint", type=Path, required=True)
    waymo_predict.add_argument("--output", type=Path, default=Path("artifacts/waymo_prediction.png"))
    waymo_predict.add_argument("--scenario-index", type=int, default=0)
    waymo_predict.add_argument("--target-id", type=int)
    preprocess_waymo = subparsers.add_parser("preprocess-waymo", help="build the sharded Waymo cache")
    preprocess_waymo.add_argument("--input", type=Path, nargs="+", required=True, help="TFRecord files, directories, or glob patterns")
    preprocess_waymo.add_argument("--split", choices=("train", "validation"), required=True)
    preprocess_waymo.add_argument("--output-dir", type=Path, required=True)
    preprocess_waymo.add_argument("--history-steps", type=int, default=11)
    preprocess_waymo.add_argument("--future-steps", type=int, default=80)
    preprocess_waymo.add_argument("--shard-size", type=int, default=1024)
    preprocess_waymo.add_argument("--max-scenarios", type=int)
    download_av2 = subparsers.add_parser("download-av2", help="download public AV2 motion scenarios")
    download_av2.add_argument("--split", choices=("train", "validation"), required=True)
    download_av2.add_argument("--output-dir", type=Path, default=Path("data/raw/av2"))
    download_av2.add_argument("--max-scenarios", type=int, required=True)
    download_av2.add_argument("--workers", type=int, default=16)
    preprocess_av2 = subparsers.add_parser("preprocess-av2", help="build the sharded AV2 cache")
    preprocess_av2.add_argument("--input", type=Path, required=True)
    preprocess_av2.add_argument("--split", choices=("train", "validation"), required=True)
    preprocess_av2.add_argument("--output-dir", type=Path, default=Path("data/processed/av2"))
    preprocess_av2.add_argument("--shard-size", type=int, default=1024)
    preprocess_av2.add_argument("--max-scenarios", type=int)
    ablation = subparsers.add_parser("run-waymo-ablation", help="run the screen or final ablation")
    ablation.add_argument("--config", type=Path, default=Path("configs/waymo_ablation.yaml"))
    ablation.add_argument("--stage", choices=("screen", "final"), required=True)
    av2_ablation = subparsers.add_parser("run-ablation", help="run a cached motion ablation")
    av2_ablation.add_argument("--config", type=Path, default=Path("configs/av2_ablation.yaml"))
    av2_ablation.add_argument("--stage", choices=("screen", "final"), required=True)
    evaluate = subparsers.add_parser("evaluate-waymo", help="evaluate one frozen ablation checkpoint")
    evaluate.add_argument("--experiment", type=Path, required=True)
    evaluate.add_argument("--official", action="store_true")
    args = parser.parse_args(argv)
    if args.command in {"smoke", "train-synthetic"}:
        _run_training(args.output, args.epochs)
        return 0
    if args.command == "train-social-synthetic":
        dataset = SyntheticDataset(count=96)
        split = int(len(dataset) * 0.8)
        _, metrics = train_social_map_model(
            list(dataset.samples[:split]), list(dataset.samples[split:]), args.output, epochs=args.epochs, modes=args.modes
        )
        print(metrics)
        return 0
    if args.command == "experiment":
        _run_experiment(args.output, args.epochs, args.count, args.history_steps, args.future_steps, args.modes)
        return 0
    if args.command == "predict-synthetic":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        from .models import HistoryGRU

        model = HistoryGRU(checkpoint["hidden_size"], checkpoint["future_steps"])
        model.load_state_dict(checkpoint["model_state"])
        from .data import make_synthetic_scenario, scenario_to_sample

        scenario = make_synthetic_scenario(args.seed, future=checkpoint["future_steps"])
        sample = scenario_to_sample(scenario, history_steps=10, future_steps=checkpoint["future_steps"])
        with torch.no_grad():
            prediction = model(torch.from_numpy(sample.history)[None]).numpy()[0]
        plot_prediction(sample, prediction, args.output, constant_velocity(sample.history, sample.future.shape[0]))
        print(f"Wrote {args.output}")
        return 0
    if args.command == "visualize-synthetic":
        from .data import make_synthetic_scenario, scenario_to_sample

        sample = scenario_to_sample(make_synthetic_scenario(args.seed), history_steps=10, future_steps=20)
        plot_scene(sample, args.output)
        print(f"Wrote {args.output}")
        return 0
    if args.command == "visualize-waymo":
        sample = _waymo_sample(args.scenario, args.scenario_index, args.target_id, args.history_steps, args.future_steps)
        plot_scene(sample, args.output)
        print(f"Wrote {args.output}")
        return 0
    if args.command == "predict-waymo":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("model_type") != "history_gru":
            raise ValueError("predict-waymo currently supports a history-GRU checkpoint.")
        from .models import HistoryGRU

        sample = _waymo_sample(args.scenario, args.scenario_index, args.target_id, history_steps=10, future_steps=checkpoint["future_steps"])
        model = HistoryGRU(checkpoint["hidden_size"], checkpoint["future_steps"])
        model.load_state_dict(checkpoint["model_state"])
        with torch.no_grad():
            prediction = model(torch.from_numpy(sample.history)[None]).numpy()[0]
        plot_prediction(sample, prediction, args.output, constant_velocity(sample.history, sample.future.shape[0]))
        print(f"Wrote {args.output}")
        return 0
    if args.command == "preprocess-waymo":
        from .waymo import preprocess_waymo_cache

        paths = _expand_waymo_inputs(args.input)
        manifest, stats = preprocess_waymo_cache(
            paths, args.output_dir, args.split, args.history_steps, args.future_steps,
            args.shard_size, args.max_scenarios,
        )
        print(json.dumps(stats, indent=2))
        print(f"Wrote {manifest}")
        return 0
    if args.command == "download-av2":
        from .av2 import download_av2

        manifest = download_av2(args.split, args.output_dir, args.max_scenarios, args.workers)
        print(f"Wrote {manifest}")
        return 0
    if args.command == "preprocess-av2":
        from .av2 import preprocess_av2_cache

        manifest, stats = preprocess_av2_cache(
            args.input, args.output_dir, args.split, args.shard_size, args.max_scenarios,
        )
        print(json.dumps(stats, indent=2))
        print(f"Wrote {manifest}")
        return 0
    if args.command in {"run-waymo-ablation", "run-ablation"}:
        from .ablation_runner import run_ablation

        output = run_ablation(args.config, args.stage)
        print(f"Wrote {output}")
        return 0
    if args.command == "evaluate-waymo":
        from .waymo_eval import evaluate_experiment

        output = evaluate_experiment(args.experiment, args.official)
        print(f"Wrote {output}")
        return 0
    return 1
