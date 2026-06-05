# CPU Baby-Zero 9x9 Smoke

## Hypothesis

The smallest local loop should prove that model evaluation, MCTS visit targets,
self-play sampling, SGF export, and one optimizer update are wired together.

## Config

- board size: 9
- rules: simplified no-capture area rules for local smoke only
- model: 16 channels, 1 residual block
- parameters: 20,029
- MCTS simulations: 8 per move
- games: 1
- max moves: 20
- batch size: 16
- train steps: 1
- learning rate: 1e-3
- c_puct: 1.5
- temperature: 1.0
- seed: 1

## Observations

First local run generated 20 positions and completed one update.

- first root value: -0.0064
- first move top visits: J7, B6, H3, B2
- policy loss: 4.252842
- value loss: 1.180121
- total loss: 5.432964

This is not meaningful棋力 yet. It is a pipe test.

Generated artifacts:

- `artifacts/cpu-demo-9x9.sgf`
- `artifacts/cpu-demo-9x9.json`
- `artifacts/visualizations/cpu-demo-9x9-dashboard.html`
- `artifacts/visualizations/cpu-demo-9x9-overview.svg`

The dashboard visualizes:

- final board and move order;
- root value by move;
- first-position search visits, priors, and values;
- policy/value loss breakdown;
- per-move search trace.

## Decision

Keep this as the CPU smoke path. Next useful step is replacing the simplified
rules backend with `sgfmill` locally or on the rented server, then implementing a
real GTP bridge for Sabaki.
