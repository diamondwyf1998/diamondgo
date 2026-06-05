# DiamondGo

DiamondGo is a learning-oriented Go AI project. The goal is to reproduce the
core AlphaZero/KataGo-style training loop while keeping every engineering choice
easy to inspect and compare.

The project is intentionally experiment-first:

- use an existing Go rules library instead of reimplementing rules;
- start with small board sizes and fast smoke tests;
- record model behavior, training curves, and engineering tricks as experiments;
- keep modules narrow enough that we can swap pieces and compare them.

## Initial Roadmap

1. Build a minimal AlphaZero loop on 9x9:
   - policy/value network;
   - MCTS guided by the network;
   - self-play data generation;
   - supervised-style update from self-play targets.
2. Add observability:
   - position snapshots;
   - policy entropy and value calibration;
   - win-rate comparisons between checkpoints;
   - ablation logs for tricks.
3. Iterate toward KataGo-like ideas:
   - rule/scoring variations through the rules adapter;
   - ownership/head experiments;
   - stronger data replay and gating;
   - search improvements and playout cap randomization.

## Layout

```text
src/diamondgo/
  config.py      Typed experiment configuration.
  rules.py       Adapter boundary for an external Go rules engine.
  model.py       Small residual policy/value network.
  mcts.py        Search data structures and expansion logic.
  selfplay.py    Self-play orchestration skeleton.
  train.py       Training entrypoint skeleton.
docs/
  roadmap.md     Technical learning path and experiment plan.
  experiments/   Experiment notes and observations.
```

## Setup

The first implementation target is Python 3.11+ with PyTorch.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

`sgfmill` is used as the first rules backend. If we outgrow it, the adapter in
`src/diamondgo/rules.py` is the intended replacement point.

## First Smoke Checks

```powershell
python -m diamondgo.train --help
python -m diamondgo.selfplay --help
```

These commands currently validate the project wiring. The next step is to turn
the skeleton into a runnable 9x9 self-play loop.

For the current local CPU smoke without installing the package:

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from diamondgo.demo_cpu import main; main()" --json
```

This writes `artifacts/cpu-demo-9x9.sgf`, which can be opened in Sabaki.
It also writes:

- `artifacts/cpu-demo-9x9.json`: compact move/search trace;
- `artifacts/visualizations/cpu-demo-9x9-dashboard.html`: standalone dashboard;
- `artifacts/visualizations/cpu-demo-9x9-overview.svg`: compact visual summary.
