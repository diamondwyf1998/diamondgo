# 13x13 B6C96 split-target archive, 2026-08-07

This archive preserves the final local/server state before shutting down the
AutoDL run:

`/root/autodl-tmp/diamondgo-runs/13x13-ce-simsched-lr0p005-margin0p6-b6c96-16w-4090d-fresh-20260803-192533`

## Saved Checkpoints

Path:

`artifacts/checkpoint-archive/13x13-b6c96-simsched-split-target-20260807`

Saved weights:

- `cycle-00020.pt` through `cycle-00220.pt`, every 20 cycles
- `cycle-00230.pt`
- `latest.pt`, corresponding to latest saved metrics cycle 232

The `.pt` files are stored with Git LFS. The archive manifest contains byte
sizes and SHA256 hashes.

## Saved Training Data

Path:

`artifacts/training-data-archive/13x13-b6c96-simsched-split-target-20260807`

Includes:

- `metrics.jsonl`, `config.json`, `run_notes.md`
- `gpu_monitor.csv`, `status_3h.log`
- `latest-cycle-00232-trace.json`, `latest-cycle-00232.sgf`
- `selfplay-recorded-every20-5games.tgz`
- `selected-cycle-records-raw.tgz`

The viewer-ready local dataset is:

`artifacts/selfplay-recorded-every20-5games-b6c96-split-target-20260807`

It contains recorded training self-play for cycles 20, 40, 60, 80, 100, 120,
140, 160, 180, 200, 220, and 230, five games per cycle.

## Latest Metrics Snapshot

Latest cycle: 232

- simulations: 800
- score komi: 9.5
- loss: 3.292196
- policy loss: 3.112983
- value loss: 0.057372
- final-board loss: 0.487361
- black win rate: 0.875
- white win rate: 0.125
- black score margin mean: 68.141
- positions per second: 9.935
- dynamic score komi reason: `black_win_rate_high`

## Local Viewer Links

- self-play subset:
  `/viewers/selfplay-catalog-viewer.html?dataset=selfplay-recorded-every20-5games-b6c96-split-target-20260807&cycle=20&game=1`
- latest status:
  `/training-status-13x13-b6c96-simsched-split-target-cycle232-20260807/index.html`
- play AI:
  `http://127.0.0.1:8787/viewers/play-ai.html?checkpoint=13x13-b6c96-simsched-split-target-cycle-00232-latest&v=training-status`
