# 2026-08-10 13x13 B6C96 split-target archive

Server was still running, so the final pre-shutdown archive was repeated and refreshed from the AutoDL/SeetaCloud run.

## Latest status

- Latest metrics cycle: 314.
- Latest periodic checkpoint available: cycle 310; `latest.pt` is cycle 314.
- Board/model: 13x13, B6C96, final-board head cross entropy, split policy target/sample temperatures.
- Simulations: 800.
- Score komi: 8.5; dynamic rule indicates next score komi 9.5 because recent black win rate remains high.
- Latest black win rate: 0.7734.
- Latest first-pass median: 172.0.
- Latest positions/s: 10.117.

## Archived locally

- Checkpoints: `artifacts/checkpoint-archive/13x13-b6c96-simsched-split-target-20260810`.
- Training data: `artifacts/training-data-archive/13x13-b6c96-simsched-split-target-20260810`.
- Recorded self-play viewer subset: `artifacts/selfplay-recorded-every20-5games-b6c96-split-target-20260810`.
- Latest status page: `artifacts/training-status-13x13-b6c96-simsched-split-target-cycle314-20260810`.

## Notes

The same sampling rule as the previous archive was extended: every 20 cycles plus the final available periodic checkpoint. For this pull, that is cycles 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 310. Each sampled self-play trace keeps games 1-5 for quick viewer loading while the raw selected traces are also preserved in the training-data archive.
