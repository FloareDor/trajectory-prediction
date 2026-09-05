# Experiment protocol

Use the experiment command to run comparable model variants:

```powershell
python -m motion_prediction experiment --output experiments/first-run --epochs 20
```

It trains the history-only and social/map multimodal models on the same
synthetic split, evaluates the constant-velocity baseline, and writes a
`summary.json` plus one subdirectory per model. The training functions also
save a loss curve in each metrics JSON.

The main knobs are explicit command-line arguments, so you can run controlled
comparisons without editing model code:

```powershell
python -m motion_prediction experiment `
  --output experiments/longer-history `
  --history-steps 20 `
  --future-steps 40 `
  --modes 6
```

Before changing code, record:

```text
Question:
Hypothesis:
One changed variable:
Expected metric change:
Expected visual change:
```

Afterward record the actual ADE/FDE, the most interesting plot, and whether the
hypothesis was supported. Keep runs with different seeds in separate output
directories; do not overwrite an earlier result.
