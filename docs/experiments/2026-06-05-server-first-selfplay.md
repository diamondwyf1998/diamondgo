# Server First Self-Play

## Hypothesis

The rented 4090D instance should run the current 9x9 baby-zero self-play and
training loop end to end, produce replay artifacts, and confirm CUDA wiring.

## Config

- server GPU: NVIDIA GeForce RTX 4090 D, 24 GB
- torch: 2.8.0+cu128
- device: cuda
- board size: 9
- rules: simplified no-capture area rules for pipeline smoke
- model: 32 channels, 2 residual blocks
- parameters: 54,461
- MCTS simulations: 32 per move
- games: 4
- max moves: 60
- generated positions: 222
- batch size: 64
- train steps: 16
- learning rate: 1e-3
- c_puct: 1.5
- temperature: 1.0
- seed: 1

## Results

- loss: 5.329867 -> 4.136791
- policy loss: 4.427664 -> 4.078767
- value loss: 0.902204 -> 0.058024

First root search:

- root value: 0.0033
- top moves: A9, B9, C9, E9, F9
- each top move had 1 visit with nearly uniform random-init priors

Generated artifacts:

- `artifacts/server-first-run/server-first-run.sgf`
- `artifacts/server-first-run/server-first-run.json`
- `artifacts/server-first-run/server-first-run-dashboard.html`
- `artifacts/server-first-run/server-first-run-overview.svg`
- `artifacts/server-first-run/server-first-run.log`

## Interpretation

This confirms the server can run the DiamondGo loop on CUDA. The result is not
meaningful棋力 yet: rules are simplified, the network is random initialized, and
MCTS is still unbatched Python. The useful signal is that self-play data flows
into policy/value optimization and produces inspectable replay artifacts.

## Next Step

Replace the simplified rules backend with the real rules adapter on the server,
then add a GTP/SGF export path that can be opened directly in Sabaki.
