# trajectory prediction for autonomous driving

## what this is

i made this to predict where cars, people, and bikes will go next in autonomous-driving scenes. the model sees 5 seconds of motion and predicts the next 6. i wanted to see if nearby agents and road-map data help.

this uses [Argoverse 2](https://argoverse.github.io/user-guide/tasks/motion_forecasting.html): tracked positions and vector maps, no images. each target is predicted on its own.

## what the numbers mean

ADE is just how far the prediction was from what actually happened. lower is better. minADE picks the closest path if the model gives more than one. macro minADE gives the three road-user types the same weight.

A7 gives 6 paths instead of 1. minFDE checks the end of the closest path. Brier-minFDE also checks if the model actually put the good path near the top. the other models only have one path, so that number is the same as minFDE for them.

the numbers are from 3 runs. the intervals compare the same scenes between two models.

## setup

i used 60,000 AV2 scenarios: 50,000 for training and 10,000 for dev. no overlap. the models trained for up to 8 epochs with seeds 17, 29, and 43.

A3 to A7 use the same transformer with the same number of parameters. the only changes are which inputs it can use and how many paths it predicts.

| ID | Model | Nearby agents | Map | Modes |
|---|---|---:|---:|---:|
| A0 | constant velocity | no | no | 1 |
| A1 | MLP | no | no | 1 |
| A2 | GRU | no | no | 1 |
| A3 | transformer | no | no | 1 |
| A4 | transformer | yes | no | 1 |
| A5 | transformer | no | yes | 1 |
| A6 | transformer | yes | yes | 1 |
| A7 | transformer | yes | yes | 6 |

## what a trajectory looks like

one dev scene. gray is the map, blue is the target history, and boxes are other agents. the green A7 path is the one minADE picked after seeing the real future.

![One AV2 scene with the map, nearby agents, and predicted trajectories](docs/figures/trajectory_example.png)

## results

![Macro minADE for all eight models](docs/figures/model_comparison.png)

| ID | Macro minADE at 6 s (m) |
|---|---:|
| A0 | 2.8714 |
| A1 | 2.3068 +/- 0.0058 |
| A2 | 2.4991 +/- 0.0177 |
| A3 | 2.2349 +/- 0.0056 |
| A4 | 2.1678 +/- 0.0418 |
| A5 | 1.9974 +/- 0.0025 |
| A6 | 1.9813 +/- 0.0092 |
| A7 | **1.1517 +/- 0.0170** |

the next figure has endpoint metrics. [full table](experiments/av2_ablation_60k/final/aggregate.md)

![Final endpoint, ranking, and miss-rate metrics](docs/figures/endpoint_metrics.png)

for A0 to A6, Brier-minFDE equals minFDE because they only predict one path. A7 gets 3.404 m minFDE and 3.943 m Brier-minFDE, so it did not rank the best endpoint very well.

![Paired minADE changes](docs/figures/component_effects.png)

the A6 to A7 row is different from the others. A7 gets 6 guesses and minADE picks the closest one after seeing what really happened. so it only says the 6 paths cover more possible futures. it does not mean A7 knew which path would happen.

| Change | Difference in macro minADE | Paired 95% interval | Result |
|---|---:|---:|---|
| A3 to A4: add neighbors | -0.0671 | [-0.0856, -0.0492] | better |
| A3 to A5: add map | -0.2374 | [-0.2709, -0.2070] | better |
| A3 to A6: add both | -0.2536 | [-0.2855, -0.2241] | better |
| A6 to A7: use six modes | -0.8296 | [-0.8607, -0.7996] | more coverage |

## what i found

the map helped more than nearby agents. both together beat the target-only transformer by 0.2536 m.

A7 has 6 guesses, so of course its best guess gets a lower minADE: **1.152 m** vs **1.981 m** for A6. that is just coverage. its first choice was worse, 3.391 m vs 2.910 m, so it was not actually better at choosing what would happen.

this is only my 60k split, not the official AV2 benchmark. [full results](experiments/av2_ablation_60k/final/aggregate.md)

## run it

run this from the project folder:

```powershell
python -m pip install -e ".[dev]"
python -m motion_prediction download-av2 --split train --max-scenarios 60000
python -m motion_prediction preprocess-av2 --input data/raw/av2/train --split train --output-dir data/processed/av2_60k --shard-size 1024 --max-scenarios 60000
python -m motion_prediction run-ablation --config configs/av2_ablation_60k.yaml --stage final
python scripts/plot_results.py
python scripts/plot_trajectory_example.py
python -m pytest -q
```

the downloader skips files already there. output goes in `experiments/av2_ablation_60k/final`.
