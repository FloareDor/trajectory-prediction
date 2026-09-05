# Learning and experimentation guide

This repository is intentionally structured as a sequence of experiments. A
model is not considered an improvement merely because its training loss went
down: compare it with the same validation scenes, inspect trajectory plots,
and write down what you expected before changing the code.

## Start with the data

Run:

```powershell
python -m motion_prediction smoke --output artifacts
```

Read `src/motion_prediction/data.py` and answer these questions in your own
notes:

1. Why is the final observed target position exactly `(0, 0)` after
   preprocessing?
2. Why does rotating into the target's heading make learning easier?
3. What do `neighbor_mask` and `map_mask` protect the model from?

Change one synthetic-scene parameter, rerun the smoke command, and inspect the
new plot. Useful parameters include curvature, number of agents, and history
length.

## Baseline experiment

The constant-velocity baseline is intentionally difficult to beat on straight
roads. It gives you a sanity check for every learned model. Read
`src/motion_prediction/baselines.py` and verify the ADE/FDE calculations by
constructing a trajectory whose future is exactly constant velocity.

## Model experiments

The models form a controlled ladder:

```text
constant velocity
  → HistoryGRU
  → SocialMapPredictor without map context
  → SocialMapPredictor with map context
  → SocialMapPredictor with multiple modes
```

For each experiment, change one factor only:

- history length: does more history help or amplify noise?
- number of neighbors: when does social context become harmful?
- map context: does it help on turns but hurt on straight roads?
- hidden size: is the model underfitting or simply over-parameterized?
- number of modes: does best-of-K improve while the most likely path worsens?

## Inspect failure cases

Metrics hide behavior. Always inspect plots for:

- turns
- crossing agents
- merges and stops
- missing or padded neighbors
- predictions that leave the road geometry

The goal is to explain *why* a prediction failed, then design the next
experiment around that explanation.

## Move to Waymo only after the synthetic path is understood

The optional adapter in `src/motion_prediction/waymo.py` converts Waymo
Scenario protobufs into the same `Scenario` structure used by the synthetic
generator. That separation lets you study model behavior without repeatedly
debugging the data format. Once a small TFRecord is available, compare a fixed
set of real scenarios with the synthetic results using the same preprocessing,
metrics, and plotting code.
