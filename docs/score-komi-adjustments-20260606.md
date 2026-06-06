# Score Komi Adjustments, 2026-06-06

This note records the live score-komi changes during the 4x64 no-komi-input
200-simulation continuation.

## Score Komi 4.5 Check

- Checked time: 2026-06-06 11:48 CST on the remote server.
- Run directory:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score4p5-200sims-noise-aug-6h-20260606`
- Latest cycle: `434`
- Black wins / White wins: `26 / 6`
- Black win rate: `81.25%`
- Mean Black score margin: `+2.531`
- Mean moves: `101.406`
- First-pass median: `71.0`
- Early first-pass `<=40`: `1/32`
- Early-pass alert: `false`
- Pass moves by color: Black `238`, White `69`

Interpretation:

- The 4.5 score komi reduced the initial skew, but the latest sample still
  exceeded the user-defined `75%` Black win-rate threshold.
- The run was therefore stopped and preserved. Its finalizer was also stopped
  so it would not spend GPU time evaluating a superseded configuration.

## Score Komi 6.5 Continuation

- New run directory:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score6p5-200sims-noise-aug-5h-20260606`
- Resume checkpoint:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score4p5-200sims-noise-aug-6h-20260606/latest.pt`
- Runtime: `300` minutes, chosen to leave time for eval and artifact backup
  before the rented server expires.
- Only score komi changes from `4.5` to `6.5`.

Unchanged settings:

| Field | Value |
| --- | --- |
| Input komi | `false` |
| Input planes | `3` |
| Komi metadata | `0.5` |
| Channels / residual blocks | `64 / 4` |
| Trainable parameters | `316,669` |
| MCTS simulations | `200` for training self-play |
| Workers | `8` |
| Games per worker / cycle | `4 / 32` |
| Max moves | `120` |
| Train steps per cycle | `64` |
| Batch size | `256` |
| Replay size | `100,000` |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| c_puct | `1.5` |
| Root Dirichlet noise | alpha `0.15`, fraction `0.25` |
| Root policy temperature | `1.1` |
| Move temperature | `1.0` through move `16`, then `0.25` |
| Data augmentation | random dihedral symmetry during training |
| Checkpoint interval | every `10` cycles |
