# Terminal reward add-ons

Date: 2026-06-06

This note records two optional experiment features. They are off by default, so old runs keep the old value target semantics.

## Features

- `--terminal-dead-stone-cleanup`
  - Before terminal scoring, remove conservatively detected obvious dead groups.
  - Current heuristic only removes an entire group when all are true:
    fewer than two solid eyes; removing it creates an empty region that does not touch the board edge; that region is bordered only by opponent stones.
  - Edge deaths, ko, seki, and complex life-and-death are deliberately not solved here. This is a diagnostic/training aid, not a full life-and-death engine.

- `--score-margin-reward-scale`
  - When disabled (`0.0`), value target remains the old pure win/loss target: `+1` or `-1`.
  - When enabled (`>0.0`), value target is bounded:
    `sign(score_margin) * (2/5 + min(abs(score_margin) ** 0.25 / 5 * scale, 3/5))`
  - With `scale=1.0`, black winning by 16 points gives black value target `0.8`; white's view is `-0.8`.
  - The margin component is capped at `3/5`, so even large wins stay within `[-1, 1]`.

## Things To Watch

- The model value head is still `tanh`, so bounded targets are a better fit than the earlier unbounded version.
- Metrics still include `value_target_min` and `value_target_max`; keep watching them so any future reward change is visible in logs.

## Wired Paths

- `demo_cpu.py`
- `batched_demo.py`
- `overnight_train.py`
- `multiworker_train.py`
- `eval_checkpoints.py`
- `tactical_eval.py`

Training and evaluation checkpoint configs record these fields, so later replay can distinguish experiment conditions.
