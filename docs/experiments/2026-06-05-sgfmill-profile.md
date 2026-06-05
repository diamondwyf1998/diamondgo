# Sgfmill Rules Profiling

## Hypothesis

Switching from the simplified smoke rules to `sgfmill` should keep the training
loop working while exposing the real bottleneck before we scale the rented
4090D.

## Environment

- GPU: NVIDIA GeForce RTX 4090 D, 24 GB
- torch: 2.8.0+cu128
- device: cuda
- rules: sgfmill

## Real-Rules Run

- board size: 9
- model: 32 channels, 2 residual blocks
- parameters: 54,461
- games: 4
- max moves: 80
- MCTS simulations: 32 per move
- generated positions: 276
- train steps: 16
- batch size: 64

Timing:

- self-play: 18.198 s
- training: 1.170 s
- artifact writing: 0.039 s
- total: 19.772 s
- throughput: 15.166 positions/s

Loss:

- loss: 5.417449 -> 4.134958
- policy loss: 4.475909 -> 4.057804
- value loss: 0.941540 -> 0.077154

## Resource Profile

Larger profile:

- games: 8
- max moves: 80
- MCTS simulations: 64 per move
- generated positions: 521
- train steps: 8
- batch size: 128

Timing:

- self-play: 68.655 s
- training: 1.141 s
- artifact writing: 0.068 s
- total: 70.129 s
- throughput: 7.589 positions/s

Short GPU-sampled profile:

- generated positions: 220
- total: 31.668 s
- throughput: 7.257 positions/s
- average GPU utilization: 4.94%
- max sampled GPU utilization: 7.0%
- average memory used: 398.8 MB
- average power: 51.5 W

## Interpretation

The current implementation is not close to saturating the 4090D. Self-play
dominates wall time, while the actual optimizer step is tiny. The GPU mostly
waits on Python MCTS, rules checks, and single-position inference.

## Decision

Do not scale by simply renting a larger GPU or increasing model size yet. The
next engineering step should be batched inference and multiple concurrent
self-play workers. A good near-term target is to collect leaf evaluations across
games and evaluate them in batches of 64-512 on GPU.

Generated artifacts:

- `artifacts/server-sgfmill-profile/server-sgfmill-profile.sgf`
- `artifacts/server-sgfmill-profile/server-sgfmill-profile.json`
- `artifacts/server-sgfmill-profile/server-sgfmill-profile-dashboard.html`
- `artifacts/server-sgfmill-profile/server-sgfmill-profile-overview.png`
- `artifacts/server-sgfmill-profile/gpu-samples.csv`
