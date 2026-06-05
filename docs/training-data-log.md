# Training Data Log

This file is the data-heavy companion to `docs/training-notes.md`. Keep raw
tables, artifact paths, checkpoint ranges, and measured values here. The notes
file should reference sections in this file instead of duplicating all numbers.

## Run And Config Timeline

| Stage | Approx checkpoints | Run/artifact family | Max moves | Komi fields observed | Notes |
| --- | ---: | --- | ---: | --- | --- |
| Early overnight | 10-400 | `overnight-9x9-20260605` | 80 | `komi=0.5` | Earlier runs before detailed behavior metrics. |
| First multiworker | 450-530 | `multiworker-9x9-100sims-90min-20260605` | 80 | `komi=0.5` | Used for initial vs previous evaluation dashboards. |
| Fastlegal/transition | 600-610 | `multiworker-9x9-100sims-fastlegal-6h-20260605` | 80 | `komi=0.5` | Center-bias and self-play samples include these checkpoints. |
| Longer games | 630-690 | `multiworker-9x9-100sims-160moves-5h-20260605` | 160 | `komi=0.5` | Max move cap raised; used by 630, 650, 660, 690 samples. |
| Optimized 120-move run | 700-856 | `multiworker-9x9-100sims-120moves-opt-v2-5h-20260605` | 120 | `komi=0.5` in sampled traces | Main color-swing run. |
| Komi split work | after source commits `3da7d3a`, `f37bd46` | source tree, smoke artifacts | 120 default | `komi=0.5`, `score_komi=6.5` | Some work done by another agent. Legacy checkpoints may not include `score_komi`. |
| Score-komi continuation | 857+ | `multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605` | 120 | model-input `komi=0.5`, scoring `score_komi=6.5` | Resumed old 0.5-komi checkpoint weights/optimizer, but rebuilt replay from new self-play. |

Relevant source commits:

- `82a2357 Use longer default self-play games`
- `3da7d3a Standardize komi and evaluation suite`
- `f37bd46 Separate model komi from scoring komi`

Relevant smoke artifacts:

- `artifacts/smoke-komi65/config.json`: older style `komi=6.5`
- `artifacts/smoke-score-komi65/config.json`: split style `komi=0.5`, `score_komi=6.5`
- Server smoke: `/root/diamondgo-score-komi-smoke/artifacts/smoke-score-komi65-sgfmill`
- Continuation run config:
  `/root/diamondgo/artifacts/multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605/config.json`

## Black Win Rate And Margin, Cycles 700-856

Source artifacts:

- `artifacts/black-margin-mean-20260605/summary.json`
- `artifacts/black-margin-median-20260605/summary.json`

All margins are Black minus White. Positive means Black leads; negative means
White leads.

| Cycle range | Black win rate | Mean Black margin |
| --- | ---: | ---: |
| 700-709 | 80.3% | +6.60 |
| 710-719 | 83.4% | +7.15 |
| 720-729 | 79.1% | +7.01 |
| 730-739 | 84.4% | +8.34 |
| 740-749 | 75.9% | +5.62 |
| 750-759 | 80.6% | +6.95 |
| 760-769 | 57.2% | +2.06 |
| 770-779 | 15.0% | -8.34 |
| 780-789 | 15.6% | -9.08 |
| 790-799 | 24.1% | -7.79 |
| 800-809 | 29.1% | -7.57 |
| 810-819 | 24.7% | -6.32 |
| 820-829 | 72.2% | +5.76 |
| 830-839 | 85.6% | +8.80 |
| 840-849 | 85.9% | +8.35 |
| 850-856 | 84.8% | +7.51 |

Overall for cycles 700-856:

- Black wins: `3048 / 5024` games, `60.7%`
- Mean Black margin over all games: `+2.09`
- Median Black margin over all games: `+4.5`
- Mean Black win margin, Black wins only: `+12.67`
- Mean White win margin, White wins only: `+14.23`
- Median Black win margin, Black wins only: `+10.5`
- Median White win margin, White wins only: `+10.5`

## Score-Komi Continuation, Cycles 857+

Source artifacts:

- Training run:
  `/root/diamondgo/artifacts/multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605`
- Resume source checkpoint:
  `/root/diamondgo/artifacts/multiworker-9x9-100sims-120moves-opt-v2-5h-20260605/latest.pt`
- Baseline copied into continuation checkpoint directory:
  `/root/diamondgo/artifacts/multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605/checkpoints/cycle-00850.pt`

This run resumes model weights and optimizer from the old 0.5-komi run, keeps
the model-input komi plane at `komi=0.5`, and changes only terminal scoring and
value labels via `score_komi=6.5`. The old replay buffer is not restored because
checkpoints do not serialize it; replay is rebuilt from new self-play.

Early measured cycles while the run was still active:

| Cycle | Black wins | White wins | Black win rate | Mean Black margin | Capture move fraction | Mean moves |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 857 | 24 | 8 | 75.0% | +3.91 | 8.82% | 93.84 |
| 858 | 18 | 14 | 56.2% | +3.00 | 9.18% | 94.94 |
| 859 | 2 | 30 | 6.2% | -19.91 | 8.37% | 94.03 |
| 860 | 3 | 29 | 9.4% | -14.47 | 7.77% | 91.25 |
| 861 | 0 | 32 | 0.0% | -20.16 | 8.16% | 89.97 |
| 862 | 1 | 31 | 3.1% | -22.12 | 8.18% | 94.38 |
| 863 | 2 | 30 | 6.2% | -17.94 | 8.06% | 94.56 |
| 864 | 2 | 30 | 6.2% | -18.47 | 8.38% | 93.59 |
| 865 | 2 | 30 | 6.2% | -14.25 | 7.50% | 92.47 |
| 866 | 0 | 32 | 0.0% | -18.38 | 8.36% | 91.16 |
| 867 | 1 | 31 | 3.1% | -20.75 | 8.29% | 95.00 |
| 868 | 0 | 32 | 0.0% | -18.69 | 9.12% | 90.16 |
| 869 | 3 | 29 | 9.4% | -12.31 | 8.36% | 94.16 |
| 870 | 3 | 29 | 9.4% | -13.94 | 7.71% | 90.78 |
| 871 | 2 | 30 | 6.2% | -16.09 | 7.75% | 94.34 |
| 872 | 0 | 32 | 0.0% | -17.91 | 8.34% | 91.09 |
| 873 | 2 | 30 | 6.2% | -15.62 | 8.41% | 94.44 |

## Center 5x5 Opening Distribution

Center-region definition: 9x9 center 5x5, rows 3-7 and columns C-G in GTP
coordinates. Pass moves are excluded. The measured window is the first 20 moves.

Source artifacts:

- `artifacts/selfplay-center-bias-600-650-690/summary.json`
- `artifacts/selfplay-center-bias-expanded-20260605/summary.json`
- `artifacts/selfplay-center-bias-after-600-20260605/summary.json` on the server

Expanded sample:

| Cycle | Black center | White center | Source max moves |
| --- | ---: | ---: | ---: |
| 100 | 36.2% | 31.8% | 80 |
| 200 | 36.7% | 43.4% | 80 |
| 300 | 40.5% | 37.0% | 80 |
| 400 | 38.5% | 48.5% | 80 |
| 450 | 52.0% | 29.5% | 80 |
| 500 | 35.0% | 66.5% | 80 |
| 530 | 46.5% | 52.0% | 80 |
| 600 | 38.0% | 49.0% | 80 |
| 650 | 67.5% | 23.5% | 160 |
| 690 | 69.0% | 29.0% | 160 |
| 710 | 68.5% | 34.5% | 120 |

After-600 sample:

| Cycle | Black center | White center | Source max moves |
| --- | ---: | ---: | ---: |
| 610 | 41.0% | 50.0% | 80 |
| 630 | 74.0% | 29.0% | 160 |
| 660 | 69.0% | 26.5% | 160 |
| 700 | 71.5% | 32.0% | 120 |
| 730 | 68.5% | 33.0% | 120 |
| 750 | 68.5% | 31.5% | 120 |

## Single-Game Showcase, Cycles 760-830

Source artifacts:

- `artifacts/selfplay-showcase-swing-760-830-20260605/index.html`
- `artifacts/selfplay-showcase-swing-760-830-20260605/viewer.html`
- `artifacts/selfplay-showcase-swing-760-830-20260605/cycle-00xxx-moves.json`
- `artifacts/selfplay-showcase-swing-760-830-20260605/sample-analysis.json`

Important display note: the old per-cycle dashboards generated from batched
self-play mix positions from three parallel games. Use `viewer.html` for
single-game qualitative inspection.

Small-sample inspection, 3 games per checkpoint:

| Checkpoint | Sample result | Center ratio, first 20 | First moves |
| --- | ---: | ---: | --- |
| 760 | B 2 / W 1 | B 70.0% / W 20.7% | F4, F4, F4 |
| 770 | B 0 / W 3 | B 50.0% / W 46.7% | E4, F4, D2 |
| 780 | B 1 / W 2 | B 33.3% / W 60.0% | H8, F4, J2 |
| 790 | B 1 / W 2 | B 30.0% / W 46.7% | C1, J2, B7 |
| 800 | B 0 / W 3 | B 25.0% / W 40.0% | J8, F8, G9 |
| 810 | B 1 / W 2 | B 37.9% / W 30.0% | pass, H7, B6 |
| 820 | B 0 / W 3 | B 32.1% / W 46.7% | E7, A4, B9 |
| 830 | B 3 / W 0 | B 60.0% / W 33.3% | G2, G2, G2 |

Checkpoint paths:

- `760`: `artifacts/multiworker-9x9-100sims-120moves-opt-v2-5h-20260605/checkpoints/cycle-00760.pt`
- `770`: `artifacts/multiworker-9x9-100sims-120moves-opt-v2-5h-20260605/checkpoints/cycle-00770.pt`
- `780`: `artifacts/multiworker-9x9-100sims-120moves-opt-v2-5h-20260605/checkpoints/cycle-00780.pt`
- `790`: `artifacts/multiworker-9x9-100sims-120moves-opt-v2-5h-20260605/checkpoints/cycle-00790.pt`
- `800`: `artifacts/multiworker-9x9-100sims-120moves-opt-v2-5h-20260605/checkpoints/cycle-00800.pt`
- `810`: `artifacts/multiworker-9x9-100sims-120moves-opt-v2-5h-20260605/checkpoints/cycle-00810.pt`
- `820`: `artifacts/multiworker-9x9-100sims-120moves-opt-v2-5h-20260605/checkpoints/cycle-00820.pt`
- `830`: `artifacts/multiworker-9x9-100sims-120moves-opt-v2-5h-20260605/checkpoints/cycle-00830.pt`

## Evaluation Results

Source artifacts:

- `artifacts/eval-multiworker-100sims-90min-latest-vs-initial/dashboard.html`
- `artifacts/eval-multiworker-100sims-90min-every-50-vs-initial/dashboard.html`
- `artifacts/eval-multiworker-100sims-90min-every-50-vs-previous/dashboard.html`

Selected earlier evaluation results:

| Match | Result |
| --- | ---: |
| cycle 537 vs initial | 19/20, 95% |
| cycle 450 vs initial | 19/20, 95% |
| cycle 500 vs initial | 19/20, 95% |
| cycle 500 vs cycle 450 | 14/20, 70% |
| cycle 537 vs cycle 500 | 16/20, 80% |
