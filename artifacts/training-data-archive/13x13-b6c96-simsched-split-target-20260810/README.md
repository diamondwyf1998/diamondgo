# 13x13 B6C96 split-target training data archive - 2026-08-10

This archive preserves the final local pull before shutting down the server.

Latest cycle: 314

Key display links:

- Status: `training-status-13x13-b6c96-simsched-split-target-cycle314-20260810/index.html`
- Latest self-play: `viewers/selfplay-catalog-viewer.html?dataset=selfplay-status-13x13-b6c96-simsched-split-target-cycle314-20260810&cycle=314&game=1`
- Recorded every-20 subset: `selfplay-recorded-every20-5games-b6c96-split-target-20260810/index.html`
- Curves: `training-curves-archive-20260619/index.html`

Included data:

- `metrics.jsonl`, `config.json`, `run_notes.md`, `gpu_monitor.csv`, `train.log`, `status_3h.log`.
- Latest full trace/SGF renamed with cycle number.
- `selected-cycle-records-raw.tgz`: raw server traces for cycles 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 310.
- The lightweight local viewer subset can be rebuilt from `selected-cycle-records-raw.tgz` with `scripts/build_recorded_selfplay_subset.py`; it is kept locally for display but not duplicated in this archive.
- `latest-summary.json` and `manifest.json`.

Latest summary: black win rate 0.7734, first-pass median 172.0, loss 3.278198.
