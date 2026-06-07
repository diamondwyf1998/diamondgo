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
| Reduced score-komi continuation | 1065+ | `multiworker-9x9-resume0p5-score2p5-100sims-120moves-1h-20260605` | 120 | model-input `komi=0.5`, scoring `score_komi=2.5` | Started after `score_komi=6.5` produced persistent White win rate above 90%. |
| Fresh no-komi-input 4x64 | pending | `multiworker-9x9-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605` | 120 | no komi input plane, scoring `score_komi=2.5` | Fresh model with deeper/wider net, root noise, late temperature drop, and dihedral augmentation. |

## Training Hyperparameters And Runtime Config

Common model architecture for the main server runs:

| Field | Value |
| --- | --- |
| Board size | `9x9` |
| Input planes | `4`: own stones, opponent stones, side-to-play plane, `komi / 10.0` plane |
| Trunk | 3x3 conv, batch norm, ReLU, then residual blocks |
| Residual blocks | `2` |
| Channels | `32` |
| Policy head | 1x1 conv to 2 channels, batch norm, ReLU, linear to `82` actions including pass |
| Value head | 1x1 conv to 1 channel, batch norm, ReLU, linear to `channels`, ReLU, linear to scalar, tanh |

## Planned Fresh Retraining Config

This is a planning record, not a completed run.

Reason for fresh retraining:

- Continuing old checkpoints while only changing `score_komi` produced large
  color skew rather than a clean correction.
- The next hypothesis is that the model should not receive komi as an input
  feature on 9x9. If komi is fixed by the experiment, the komi plane may make
  color/score adaptation harder to interpret.
- Turning off `input_komi` changes the model input from 4 planes to 3 planes, so
  this should be a fresh run instead of a checkpoint resume.

Planned core settings:

| Field | Planned value |
| --- | --- |
| Board size | `9x9` |
| Input planes | `3`: own stones, opponent stones, side-to-play plane |
| `input_komi` | `False`, via `--no-input-komi` |
| Residual blocks | `4` |
| Channels | planned discussion item: `32` depth-only control vs `64` wider model |
| Scoring komi | planned discussion item; actual launched value was `2.5` |
| Max moves | likely `120` |
| MCTS simulations | likely `100` |
| Workers | tune to new CPU host; expected target `16` workers on a 16-core EPYC host, then benchmark |

Working-tree support currently observed:

| Feature | Status |
| --- | --- |
| Optional komi input plane | Added as `input_komi` in rules/config/training/eval paths |
| 3-plane model stem | `PolicyValueNet(..., input_planes=3)` selected when `input_komi=False` |
| Residual block count override | Existing CLI flag `--residual-blocks`; planned value must be passed as `4` unless default is changed |
| Root Dirichlet noise | CLI plumbing added as `--root-dirichlet-alpha` and `--root-noise-fraction` |
| Root policy temperature | CLI plumbing added as `--root-policy-temperature` |
| Dihedral augmentation | CLI plumbing added as `--augment-dihedral` |

Open launch decisions:

- Choose scoring komi for the fresh 9x9 run.
- Decide whether `channels` remains `32` or increases to `64`.
- Decide root exploration settings. Current defaults keep root noise off.
- Run a short smoke test before committing to a long server run:
  `--residual-blocks 4 --no-input-komi`, a few cycles, then tactical probes and
  a small self-play viewer sample.

Launch deviation to remember:

- The launched run used `channels=64`, not a depth-only `channels=32` control.
- This should have been reported explicitly in chat before launch because it
  changes model size by far more than the `2 -> 4` residual-block change alone.

Common self-play/training settings for the main multi-worker server runs:

| Field | Value |
| --- | --- |
| Rules backend | `sgfmill` |
| MCTS simulations per move | `100` |
| `c_puct` | `1.5` |
| Move sampling temperature | `1.0` |
| Root Dirichlet noise | Not used in current batched/multiworker self-play |
| Workers | `8` |
| Games per worker per cycle | `4` |
| Games per cycle | `32` |
| Train steps per cycle | `64` |
| Batch size | `256` |
| Replay buffer cap | `50,000` positions |
| Optimizer | `AdamW` |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| Checkpoint interval | every `10` cycles |
| Device | `cuda` |
| Seed | `1` |

New experimental knobs added after the komi-skew investigation:

| Field | Meaning | Legacy default | Fresh 4x64 setting |
| --- | --- | ---: | ---: |
| `input_komi` | Whether `komi / 10.0` is appended as an input plane | `true` | `false` |
| `root_dirichlet_alpha` | Dirichlet alpha for root exploration noise | `0.0` | `0.15` |
| `root_noise_fraction` | Fraction of root prior replaced by Dirichlet noise | `0.0` | `0.25` |
| `root_policy_temperature` | Softens/sharpens raw policy priors before root expansion | `1.0` | `1.1` |
| `temperature_moves` | Number of opening moves using main sampling temperature | `0` | `16` |
| `late_temperature` | Sampling temperature after `temperature_moves` | `1.0` | `0.25` |
| `augment_dihedral` | Random rotations/reflections during replay training | `false` | `true` |

## Model Size Comparison

Parameter counts measured from `PolicyValueNet(9, ModelConfig(...))`:

| Model | Input planes | Channels | Residual blocks | Trainable parameters | Ratio vs old 32x2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old main model | 4 | 32 | 2 | `54,461` | `1.00x` |
| Depth-only control target | 3 | 32 | 4 | about `91k` | about `1.7x` |
| Current fresh 4x64 | 3 | 64 | 4 | `316,669` | `5.82x` |

Important interpretation:

- The current fresh run changed both depth and width.
- Channel width increases convolution parameters roughly with `channels^2`, so
  `32 -> 64` is much larger than a linear `2x` change.
- The intended depth-only comparison should use `32 channels x 4 residual
  blocks`, with the same no-komi-input/root-noise/temperature/augmentation
  tricks, before claiming the benefit of the larger `64x4` network.

Run-specific differences:

| Run/artifact family | Time limit | Resume | Max moves | Komi/scoring | Notes |
| --- | ---: | --- | ---: | --- | --- |
| `multiworker-9x9-100sims-90min-20260605` | 90 min | none | 80 | `komi=0.5` | First main multiworker run. |
| `multiworker-9x9-100sims-160moves-5h-20260605` | 300 min | continuation from earlier sequence | 160 | `komi=0.5` | Longer-game experiment. |
| `multiworker-9x9-100sims-120moves-opt-v2-5h-20260605` | 300 min | continuation from earlier sequence | 120 | `komi=0.5` | Main 700-856 color-swing run. |
| `multiworker-9x9-komi6p5-100sims-120moves-2h-20260605` | 120 min | none | 120 | old style `komi=6.5` | Brief fresh run, stopped after realizing `komi` changes model input. |
| `multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605` | 120 min, then queued +120 min extension, stopped early at about 21:52 CST | `multiworker-9x9-100sims-120moves-opt-v2-5h-20260605/latest.pt`; extension resumes the same run's `latest.pt` | 120 | model-input `komi=0.5`, scoring `score_komi=6.5` | Stopped because White self-play win rate stayed above 90%. Old replay buffer is not restored. The extension also rebuilt replay after its resume boundary. |
| `multiworker-9x9-resume0p5-score2p5-100sims-120moves-1h-20260605` | stopped early at cycle 1080 | `multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605/latest.pt` | 120 | model-input `komi=0.5`, scoring `score_komi=2.5` | Follow-up to test whether reducing scoring komi pulls back the White win-rate skew; stopped because the skew remained high. |
| `multiworker-9x9-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605` | 120 min planned | none | 120 | no model-input komi plane, scoring `score_komi=2.5` | Fresh-start diagnostic run; scripts are in `tools/server/`. |

Evaluation and probe defaults:

| Tool | Main settings |
| --- | --- |
| `eval_suite.py` | checkpoint step tiers `50,200,500`; opponents `initial,previous`; `20` games per match; `100` MCTS simulations; `max_moves=120`; `2` sample SGFs per match; device `cuda` |
| `eval_checkpoints.py` | alternates candidate color, opening sampling for first `6` moves, then visit-count argmax; uses checkpoint serialized config unless command-line overrides apply |
| `tactical_eval.py` | fixed tactical probes; usually `100` MCTS simulations; reports target top1/top3/rank |

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
- Extension launcher:
  `/root/diamondgo/extend_resume_scorekomi65_2h.sh`
- Extension log:
  `/root/diamondgo/artifacts/multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605/train-extension-2h.log`
- Extension cutoff log:
  `/root/diamondgo/artifacts/multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605/cutoff-after-1h.log`
- Final extended evaluation output:
  `/root/diamondgo/artifacts/eval-suite-resume0p5-score6p5-100sims-120moves-extended-20260605`

This run resumes model weights and optimizer from the old 0.5-komi run, keeps
the model-input komi plane at `komi=0.5`, and changes only terminal scoring and
value labels via `score_komi=6.5`. The old replay buffer is not restored because
checkpoints do not serialize it; replay is rebuilt from new self-play. A
two-hour extension was queued after the first segment: it waits for the current
training PID to finish, resumes the run's latest checkpoint, trains for another
120 minutes, then runs the final eval suite and tactical probes. After the
extension started, it had about 91 minutes remaining at 21:41 CST, so a cutoff
watcher was added at 21:42 CST to stop training about one hour later and run the
same final eval/tactical probes. Before that cutoff time, the run was stopped
manually because White self-play win rate remained above 90%.

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

Late cycles before stopping the `score_komi=6.5` continuation:

| Cycle | Black wins | White wins | Black win rate | White win rate | Mean Black margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1059 | 3 | 29 | 9.4% | 90.6% | -15.50 |
| 1060 | 2 | 30 | 6.2% | 93.8% | -13.91 |
| 1061 | 1 | 31 | 3.1% | 96.9% | -12.78 |
| 1062 | 2 | 30 | 6.2% | 93.8% | -12.75 |
| 1063 | 3 | 29 | 9.4% | 90.6% | -13.84 |
| 1064 | 1 | 31 | 3.1% | 96.9% | -14.47 |

## Score-Komi 2.5 Continuation, Cycles 1065+

Source artifacts:

- Training script:
  `tools/server/run_resume_scorekomi25_1h.sh`
- Server launcher:
  `/root/diamondgo/run_resume_scorekomi25_1h.sh`
- Training run:
  `/root/diamondgo/artifacts/multiworker-9x9-resume0p5-score2p5-100sims-120moves-1h-20260605`
- Finalizer script:
  `tools/server/finalize_scorekomi25_1h.sh`
- Final evaluation output:
  `/root/diamondgo/artifacts/eval-suite-resume0p5-score2p5-100sims-120moves-1h-20260605`
- Final tactical output:
  `/root/diamondgo/artifacts/tactical-resume0p5-score2p5-100sims-120moves-1h-20260605`

This follow-up resumes the latest weights after stopping the `score_komi=6.5`
run. It keeps the model-input komi plane at `komi=0.5` and changes only terminal
scoring/value labels to `score_komi=2.5`.

Initial cycles after the switch still show strong White skew; this is expected
to be slow to correct because the weights come from the White-favored state.
The run was stopped at cycle `1080`; checkpoints were preserved, including
`cycle-01070.pt`, `cycle-01080.pt`, and `latest.pt`.

| Cycle | Black wins | White wins | Black win rate | White win rate | Mean Black margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1065 | 1 | 31 | 3.1% | 96.9% | -11.41 |
| 1066 | 3 | 29 | 9.4% | 90.6% | -10.84 |
| 1067 | 2 | 30 | 6.2% | 93.8% | -11.66 |
| 1068 | 5 | 27 | 15.6% | 84.4% | -8.44 |
| 1069 | 2 | 30 | 6.2% | 93.8% | -12.22 |
| 1070 | 3 | 29 | 9.4% | 90.6% | -15.22 |
| 1071 | 1 | 31 | 3.1% | 96.9% | -11.19 |
| 1072 | 3 | 29 | 9.4% | 90.6% | -10.25 |
| 1073 | 1 | 31 | 3.1% | 96.9% | -14.09 |
| 1074 | 1 | 31 | 3.1% | 96.9% | -12.09 |
| 1075 | 4 | 28 | 12.5% | 87.5% | -9.06 |
| 1076 | 2 | 30 | 6.2% | 93.8% | -15.62 |
| 1077 | 8 | 24 | 25.0% | 75.0% | -6.06 |
| 1078 | 5 | 27 | 15.6% | 84.4% | -9.91 |
| 1079 | 5 | 27 | 15.6% | 84.4% | -10.19 |
| 1080 | 2 | 30 | 6.2% | 93.8% | -12.97 |

## Fresh No-Komi-Input 4x64 Run

Source artifacts:

- Training script:
  `tools/server/run_fresh_nokomi_4x64_score2p5_noise_aug_2h.sh`
- Finalizer script:
  `tools/server/finalize_fresh_nokomi_4x64_score2p5_noise_aug_2h.sh`
- Planned server output:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605`
- Planned eval output:
  `/root/diamondgo/artifacts/eval-suite-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605`
- Planned tactical output:
  `/root/diamondgo/artifacts/tactical-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605`

Planned config:

| Field | Value |
| --- | --- |
| Fresh start | `true` |
| Board size | `9x9` |
| Rules backend | `sgfmill` |
| Model input planes | `3`: own stones, opponent stones, side-to-play |
| `komi` config value | `0.5`, retained only for metadata because `input_komi=false` |
| `score_komi` | `2.5` |
| Channels | `64` |
| Residual blocks | `4` |
| Simulations per move | `100` |
| Workers / games per worker | `8 / 4` |
| Games per cycle | `32` |
| Max moves | `120` |
| Train steps per cycle | `64` |
| Batch size | `256` |
| Replay size | `100,000` |
| Optimizer | `AdamW` |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| `c_puct` | `1.5` |
| Opening temperature | `1.0` for first `16` moves |
| Late temperature | `0.25` |
| Root Dirichlet noise | alpha `0.15`, fraction `0.25` |
| Root policy temperature | `1.1` |
| Replay augmentation | random dihedral rotations/reflections |
| Time limit | `120` minutes |
| Checkpoint interval | every `10` cycles |

Initial server status:

- Started at about `2026-06-05 22:47 CST`.
- Training PID: `399061`
- Finalizer PID: `399713`
- Server scripts were converted to LF line endings before launch because the
  Windows checkout stored them with CRLF.
- Stopped at about `2026-06-05 23:51 CST` so the user could prepare an
  overnight training deployment. The finalizer was stopped first, so no
  automatic eval/tactical probes were launched by this pause.
- Preserved checkpoints include `cycle-00010.pt`, `cycle-00020.pt`,
  `cycle-00030.pt`, `cycle-00040.pt`, `cycle-00050.pt`, `cycle-00060.pt`, and
  `latest.pt`.

Early measured cycles:

| Cycle | Black wins | White wins | Black win rate | White win rate | Mean Black margin | Mean moves | Ended by pass | Ended by max moves |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11 | 21 | 34.4% | 65.6% | +3.16 | 71.84 | 27 | 5 |
| 2 | 19 | 13 | 59.4% | 40.6% | +1.00 | 99.44 | 26 | 6 |
| 62 | 24 | 8 | 75.0% | 25.0% | -0.56 | 108.34 | 24 | 8 |
| 63 | 25 | 7 | 78.1% | 21.9% | +0.97 | 101.12 | 26 | 6 |
| 64 | 27 | 5 | 84.4% | 15.6% | +3.19 | 103.34 | 25 | 7 |
| 65 | 23 | 9 | 71.9% | 28.1% | +2.31 | 105.12 | 25 | 7 |
| 66 | 26 | 6 | 81.2% | 18.8% | +1.38 | 105.56 | 24 | 8 |

## Fresh No-Komi-Input 4x64 Checkpoint 60 Showcase

Source artifacts:

- Local self-play viewer:
  `artifacts/selfplay-showcase-fresh-nokomi-4x64-cycle60-20260606/viewer.html`
- Local self-play data:
  `artifacts/selfplay-showcase-fresh-nokomi-4x64-cycle60-20260606/cycle-00060-moves.json`
- Server source trace:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605/latest-cycle-trace.json`
- Local play checkpoint:
  `artifacts/play-ai-checkpoints/fresh-nokomi-4x64-cycle-00060.pt`
- Browser play page:
  `artifacts/viewers/play-ai.html`

Configuration:

| Field | Value |
| --- | --- |
| Checkpoint | `cycle-00060` |
| Input komi | `false` |
| Channels / residual blocks | `64 / 4` |
| Score komi | `2.5` |
| Self-play simulations | `100` |
| Max moves | `120` |

Latest trace summary:

| Metric | Value |
| --- | ---: |
| Games | `32` |
| Positions | `3378` |
| Black wins | `26` |
| White wins | `6` |
| Black win rate | `81.2%` |

The local browser play server was updated to read `input_komi` from the
checkpoint config, so no-komi checkpoints encode positions with 3 input planes.
The same server currently serves the `cycle-00060` checkpoint on
`http://127.0.0.1:8787/viewers/play-ai.html`.

## Cross-Generation Evaluation, Fresh Cycle 40

Candidate:

- `fresh-nokomi-4x64-cycle-00040`
- Server checkpoint:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605/checkpoints/cycle-00040.pt`

Evaluation settings:

| Field | Value |
| --- | --- |
| Games per match | `10` |
| Candidate colors | `5` Black games, `5` White games |
| MCTS simulations | `100` |
| Max moves | `120` |
| Opening sampling | first `6` moves sampled from visit distribution, then max-visit |
| Device | server `cuda` |

Source artifacts:

- `artifacts/crossgen-fresh-nokomi40-vs-overnight-20260606/dashboard.html`
- `artifacts/crossgen-fresh-nokomi40-vs-overnight-20260606/results.json`
- `artifacts/crossgen-fresh-nokomi40-vs-old-300-500-20260606/dashboard.html`
- `artifacts/crossgen-fresh-nokomi40-vs-old-300-500-20260606/results.json`

Results:

| Opponent | Source line | Note | Candidate wins | Black wins | White wins |
| --- | --- | --- | ---: | ---: | ---: |
| `cycle-00040` | `overnight-9x9-20260605` | same index | `10/10` | `5/5` | `5/5` |
| `cycle-00050` | `overnight-9x9-20260605` | nearest available to requested `40 * 1.3 = 52` | `10/10` | `5/5` | `5/5` |
| `cycle-00080` | `overnight-9x9-20260605` | `40 * 2` | `9/10` | `4/5` | `5/5` |
| `cycle-00200` | `overnight-9x9-20260605` | `40 * 5` | `8/10` | `4/5` | `4/5` |
| `cycle-00300` | `overnight-9x9-20260605` | old generation cycle 300 | `7/10` | `4/5` | `3/5` |
| `cycle-00400` | `overnight-9x9-20260605` | old generation cycle 400 | `10/10` | `5/5` | `5/5` |
| `cycle-00500` | `multiworker-9x9-100sims-90min-20260605` | overnight line stops at `cycle-00410` | `10/10` | `5/5` | `5/5` |

Interpretation caveats:

- These are quick 10-game matches, useful for orientation but noisy.
- The candidate differs from the old generation in multiple variables:
  architecture, input features, root noise, sampling schedule, augmentation,
  and scoring komi.
- Because the current fresh run is already Black-skewed in self-play, color
  split should always be inspected alongside total win rate.

## Unfinished 2-Layer Baseline Questions

The earlier `32 channels x 2 residual blocks` line should not be treated as
fully explored. It underperformed in the latest continuation experiments, but
those runs also changed scoring labels midstream and inherited skewed weights.

Open follow-ups:

| Question | Why it matters | Suggested test |
| --- | --- | --- |
| Can 2-layer improve with more search? | The current main 2-layer runs used `100` simulations per move; search quality may be the bottleneck rather than capacity. | Fresh 2-layer run with matched rules/input setup and higher MCTS, such as `200` or `400` simulations. |
| How much do advanced tricks help 2-layer? | If a small model plus better self-play tricks is enough for 9x9 baby cases, it is easier to interpret than a larger model. | Add root noise, late temperature drop, dihedral augmentation, and possibly better replay/eval scheduling to the 2-layer line. |
| Is 4x64 actually stronger, or just different? | The current 4x64 fresh run changes several variables at once. | PK 2-layer vs 4x64 at matched wall-clock, matched self-play positions, and matched checkpoint intervals. |
| What should be matched for PK? | Same-time and same-data comparisons answer different questions. | Record both: same training time and same total positions/train steps. |

Planned PK outputs should include:

- win rate by color
- same checkpoint step tiers as the current eval suite: `50,200,500`
- at least `20` games per match for quick checks, larger if a candidate looks close
- two rendered sample games per match for human review
- root value and top search candidates in the dashboard

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

Qualitative observation index:

- User observation: during the White-favored phase around `770-810`, Black's
  stones can look more scattered and easier for White to surround in large
  regions.
- User observation: when Black's win rate recovered around `820+`, Black's
  style looked more stable and less dispersed. This may explain part of the
  win-rate reversal, but it has not yet been quantified.
- Candidate comparison checkpoints for this observation: `800`, `810`, `820`,
  `830`.

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

## Tactical Probes, Cycles 760-830

Source artifacts:

- `artifacts/tactical-swing-760-830-20260605/tactical_results.json`
- `artifacts/tactical-swing-760-830-20260605/tactical_report.md`
- `artifacts/tactical-swing-760-830-20260605/casebook.html`

Probe interpretation:

- Capture and atari-defense cases are positive tests: higher top1/top3 hit
  counts are better.
- Fill-eye and self/dead-shape cases are negative tests: top1/top3 counts mean
  the model chose or highly ranked a bad move, so lower is better.
- These probes use 100 MCTS simulations per move.
- The rendered casebook is the visual source of truth for the probe positions.
  Use it when checking whether a hand-built case is reasonable before trusting
  aggregate statistics.

Category summary by checkpoint:

| Cycle | Capture top1/top3 | Atari defense top1/top3 | Fill-eye bad top1/top3 | Self/dead bad top1/top3 |
| ---: | ---: | ---: | ---: | ---: |
| 760 | 1/1 of 4 | 1/1 of 4 | 0/2 of 2 | 0/1 of 2 |
| 770 | 1/2 of 4 | 1/1 of 4 | 2/2 of 2 | 0/0 of 2 |
| 780 | 1/2 of 4 | 1/1 of 4 | 1/1 of 2 | 0/0 of 2 |
| 790 | 1/2 of 4 | 1/1 of 4 | 1/1 of 2 | 0/0 of 2 |
| 800 | 1/1 of 4 | 1/1 of 4 | 1/1 of 2 | 0/0 of 2 |
| 810 | 1/1 of 4 | 2/2 of 4 | 1/1 of 2 | 0/0 of 2 |
| 820 | 1/2 of 4 | 0/1 of 4 | 1/1 of 2 | 0/1 of 2 |
| 830 | 1/2 of 4 | 1/2 of 4 | 0/1 of 2 | 0/1 of 2 |

Per-case summary across the eight checkpoints:

| Case | Category | Hit count |
| --- | --- | ---: |
| `black_capture_one_stone` | capture | good top1/top3 `0/3 of 8` |
| `white_capture_one_stone` | capture | good top1/top3 `8/8 of 8` |
| `black_capture_two_stones` | capture | good top1/top3 `0/0 of 8` |
| `white_capture_two_stones` | capture | good top1/top3 `0/2 of 8` |
| `black_escape_atari` | atari defense | good top1/top3 `0/0 of 8` |
| `white_escape_atari` | atari defense | good top1/top3 `0/0 of 8` |
| `black_capture_to_defend` | atari defense | good top1/top3 `6/7 of 8` |
| `white_capture_to_defend` | atari defense | good top1/top3 `2/3 of 8` |
| `black_avoid_filling_own_eye` | fill eye | bad top1/top3 `6/8 of 8` |
| `white_avoid_filling_own_eye` | fill eye | bad top1/top3 `1/2 of 8` |
| `black_avoid_self_atari_edge` | self/dead shape | bad top1/top3 `0/3 of 8` |
| `white_avoid_self_atari_edge` | self/dead shape | bad top1/top3 `0/0 of 8` |

## Tactical Probes, Fresh No-Komi 4x64 Cycles 10-60

Source artifacts:

- `artifacts/tactical-fresh-nokomi-4x64-cycle10-60-20260606/tactical_results.json`
- `artifacts/tactical-fresh-nokomi-4x64-cycle10-60-20260606/tactical_report.md`
- `artifacts/tactical-fresh-nokomi-4x64-cycle10-60-20260606/casebook.html`

Probe settings:

| Field | Value |
| --- | --- |
| Checkpoints | `10,20,30,40,50,60` |
| Simulations | `100` |
| Cases | `12` from the rendered tactical casebook |
| Positive categories | `capture`, `atari_defense`; higher top1/top3 is better |
| Negative categories | `fill_eye`, `self_atari_or_dead`; lower bad top1/top3 is better |

Category summary by checkpoint:

| Cycle | Capture top1/top3 | Atari defense top1/top3 | Fill-eye bad top1/top3 | Self/dead bad top1/top3 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0/0 of 4 | 0/0 of 4 | 1/1 of 2 | 0/0 of 2 |
| 20 | 0/0 of 4 | 1/1 of 4 | 1/1 of 2 | 0/0 of 2 |
| 30 | 1/1 of 4 | 1/1 of 4 | 1/1 of 2 | 0/0 of 2 |
| 40 | 1/1 of 4 | 1/1 of 4 | 1/1 of 2 | 0/0 of 2 |
| 50 | 1/1 of 4 | 1/1 of 4 | 1/1 of 2 | 0/0 of 2 |
| 60 | 1/1 of 4 | 0/1 of 4 | 1/1 of 2 | 0/0 of 2 |

Observed tactical pattern:

- Capture and atari-defense behavior is still weak in this small hand-built
  probe set. By cycle `60`, only `1/4` capture targets are top1/top3, and
  `1/4` atari-defense targets are top3 but none are top1.
- Fill-eye remains a persistent issue: in all tested checkpoints, `1/2`
  fill-eye bad moves are ranked top1/top3.
- Self-atari/dead-shape bad moves do not reach top1/top3 in this run, though
  one appears in top10 in the raw results.

## Partial Eval Before 200-Sim Overnight Continuation

Source artifacts:

- Server eval directory:
  `/root/diamondgo/artifacts/eval-suite-fresh-nokomi-4x64-score2p5-100sims-paused-20260606`
- Example dashboard generated before the eval was stopped:
  `/root/diamondgo/artifacts/eval-suite-fresh-nokomi-4x64-score2p5-100sims-paused-20260606/step-00010-vs-initial/dashboard.html`

Settings:

| Field | Value |
| --- | --- |
| Candidate line | fresh no-komi-input `4x64`, paused at cycle `66` |
| Eval simulations | `100` |
| Games per match | `20` |
| Max moves | `120` |
| Opponent completed before stop | `initial` |

The full standard eval suite was started first, but it was stopped after the
initial-opponent pass so the 16-hour overnight run could begin. Results below
are therefore a partial orientation check, not the complete 50/200/500 tier
suite.

| Candidate | Result vs initial | Candidate Black wins | Candidate White wins | Seconds |
| --- | ---: | ---: | ---: | ---: |
| cycle `10` | `16/20` (`80%`) | `10` | `6` | `136.449` |
| cycle `20` | `12/20` (`60%`) | `7` | `5` | `139.344` |
| cycle `30` | `13/20` (`65%`) | `8` | `5` | `139.809` |
| cycle `40` | `18/20` (`90%`) | `9` | `9` | `140.006` |
| cycle `50` | `18/20` (`90%`) | `8` | `10` | `139.089` |
| cycle `60` | `9/20` (`45%`) | `2` | `7` | `102.902` |
| latest/cycle `66` | `9/20` (`45%`) | `1` | `8` | `115.137` |

Interpretation update:

- The low `cycle 60` and `latest/cycle 66` scores against the initial model are
  not evidence that the new model is weaker than the initial net in ordinary
  play.
- Manual/user inspection identified the failed games as early-pass / premature
  termination failures. A quick parse of `results.json` confirms that losing
  games include candidate passes around the early middle game, commonly in the
  `20s-40s` move range, followed by an opponent pass that ends the game.
- For `cycle 60`, all `11` losses include pass sequences; examples include
  candidate Black passing at moves `25,27,29,31,33`, and candidate White passing
  at moves such as `22,24,26,32`.
- For latest/cycle `66`, all `11` losses also include pass sequences; examples
  include candidate Black passing at moves `29,31,33,35,39`, and candidate
  White passing repeatedly in several losses.
- Read this eval as a pass-policy / termination pathology. It should not be
  used as a clean strength ranking unless early pass is disabled, pass is
  heavily filtered before a minimum move count, or the eval rules handle
  premature pass more robustly.

Instrumentation added before the overnight run:

- Training `game_behavior` metrics now include first-pass, second-pass,
  terminal double-pass, pass-by-color, first-pass `<=20/40/60`, and an
  `early_pass_alert` field.
- Evaluation reports and dashboards now display first-pass median,
  first-pass `<=40`, terminal double-pass median, and an early-pass alert.
- This is an observation change only: legal moves, terminal rules, scoring, and
  value targets are unchanged.

## 200-Sim Overnight Continuation Configuration

Run directory:

- `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-200sims-noise-aug-16h-20260606`

Resume source:

- `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605/latest.pt`

Configuration:

| Field | Value |
| --- | --- |
| Fresh start | `false` |
| Resume semantics | model, optimizer, cycle, position, and train-step counters resume; replay buffer is rebuilt in the new output directory |
| Starting checkpoint | fresh no-komi-input `4x64` latest at cycle `66` |
| Input komi | `false` |
| Input planes | `3`: own stones, opponent stones, side-to-play |
| Komi metadata | `0.5` |
| Score komi | `2.5` |
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
| Time limit | `960` minutes (`16` hours) |

Expected interpretation:

- This isolates the MCTS-search increase more than the previous architecture
  change did: the model is the same fresh no-komi-input `4x64` network, but
  self-play search doubles from `100` to `200` simulations per move.
- Because the new output directory does not restore replay, early continuation
  cycles are trained only on new 200-simulation games, not the old 100-sim
  replay buffer.
- The finalizer uses the standard `100`-simulation eval/tactical checks for
  comparability with earlier dashboards; the training self-play itself is the
  part changed to `200`.

## 200-Sim Continuation Cycle 403 Snapshot

Source artifacts:

- Server training directory:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-200sims-noise-aug-16h-20260606`
- Local rendered self-play and curves:
  `artifacts/selfplay-showcase-fresh-nokomi-4x64-200sims-cycle400-20260606/`
- Main curve dashboard:
  `artifacts/selfplay-showcase-fresh-nokomi-4x64-200sims-cycle400-20260606/training-curves.html`
- Replay viewer:
  `artifacts/selfplay-showcase-fresh-nokomi-4x64-200sims-cycle400-20260606/viewer.html?cycle=403&game=1`

Checkpoint/version note:

- `latest-cycle-trace.json` and `metrics.jsonl` are at cycle `403`.
- The newest numbered checkpoint file pulled locally is
  `cycle-00400.pt`; interpret the rendered self-play as the latest trace,
  not necessarily as the exact numbered checkpoint file.

Latest cycle metrics:

| Field | Value |
| --- | ---: |
| Cycle | `403` |
| Games | `32` |
| Black wins / White wins | `26 / 6` |
| Black win rate | `81.25%` |
| Mean Black score margin | `+2.406` |
| Mean absolute score margin | `3.844` |
| Mean moves | `98.875` |
| Ended by pass / max moves | `29 / 3` |
| Early first-pass `<40` | `7/32` (`21.88%`) |
| Pass move fraction | `11.13%` |
| Capture move fraction | `11.47%` |
| Captured stones | `693` |
| Loss / policy / value | `1.3428 / 1.1660 / 0.1768` |
| Policy entropy mean | `0.5819` |
| Positions/sec | `28.682` |

Observed interpretation:

- Search is deeper (`200` sims) and the training losses are much lower than
  the early fresh run, but the self-play color distribution remains strongly
  Black-skewed in this latest snapshot.
- The previous early-pass failure mode is still visible as a behavior to
  monitor, though this cycle is not dominated by immediate premature endings:
  the latest early first-pass `<40` rate is `21.88%`, below the alert threshold.
- Timing curves continue to show legal-action generation and tree selection as
  major CPU-side costs during self-play.

## Tactical Probes, 200-Sim Continuation Cycles 100-400

Source artifacts:

- Server output:
  `/root/diamondgo/artifacts/tactical-fresh-nokomi-4x64-200sims-cycle100-400-20260606`
- Local rendered casebook:
  `artifacts/tactical-fresh-nokomi-4x64-200sims-cycle100-400-20260606/casebook.html`
- Raw results:
  `artifacts/tactical-fresh-nokomi-4x64-200sims-cycle100-400-20260606/tactical_results.json`

Settings:

| Field | Value |
| --- | --- |
| Checkpoint line | `score_komi=2.5`, no-komi-input `4x64`, 200-sim continuation |
| Tested checkpoints | `100, 150, 200, 250, 300, 350, 400` |
| Tactical cases | `10` retained rendered casebook probes |
| Probe simulations | `100` |
| Positive categories | `capture`, `atari_defense`; higher top1/top3 is better |
| Negative categories | `fill_eye`, `self_atari_or_dead`; lower bad top1/top3 is better |

Note: `black_capture_two_stones` and `white_capture_two_stones` were removed
from this summary after inspection because they were not suitable as basic
capture probes.

Summary:

| Cycle | Capture top1/top3 | Atari defense top1/top3 | Fill-eye bad top1/top3 | Self/dead bad top1/top3 |
| ---: | ---: | ---: | ---: | ---: |
| `100` | `1/1 of 2` | `0/1 of 4` | `0/0 of 2` | `0/0 of 2` |
| `150` | `0/1 of 2` | `1/3 of 4` | `0/0 of 2` | `0/0 of 2` |
| `200` | `0/1 of 2` | `3/4 of 4` | `0/0 of 2` | `0/0 of 2` |
| `250` | `0/1 of 2` | `4/4 of 4` | `1/1 of 2` | `0/1 of 2` |
| `300` | `0/1 of 2` | `4/4 of 4` | `0/0 of 2` | `0/1 of 2` |
| `350` | `1/1 of 2` | `4/4 of 4` | `0/0 of 2` | `0/1 of 2` |
| `400` | `0/1 of 2` | `4/4 of 4` | `0/0 of 2` | `1/1 of 2` |

Observed interpretation:

- The model appears to have learned basic atari defense by the middle of this
  200-sim run. From cycle `250` onward, all four atari-defense targets are
  top1/top3.
- The earlier fill-eye habit is much reduced in this fixed probe set. The only
  recurrence among tested checkpoints is cycle `250`.
- Active one-stone capture is still inconsistent. The target reaches top3 in
  all tested cycles, but is top1 only at cycles `100` and `350`.
- Self-atari/dead-shape avoidance is not monotonic: it is clean at cycles
  `100-200`, but the bad black edge/corner move is top1 again at cycle `400`.

## Tactical Level 2 Probes, 200-Sim Continuation Cycles 100-410

Source artifacts:

- Server output:
  `/root/diamondgo/artifacts/tactical-level2-fresh-nokomi-4x64-200sims-cycle100-410-20260606`
- Local rendered casebook:
  `artifacts/tactical-level2-fresh-nokomi-4x64-200sims-cycle100-410-20260606/casebook.html`
- Raw results:
  `artifacts/tactical-level2-fresh-nokomi-4x64-200sims-cycle100-410-20260606/tactical_results.json`

Design:

| Category | Cases | Intended skill |
| --- | ---: | --- |
| `life_death` | `4` | Kill hard-surrounded straight-three and bent-three eye spaces by playing the vital point |
| `double_atari` | `2` | Play the shared point that puts two enemy stones into atari |
| `second_line_atari` | `2` | Give third-line atari to a second-line stone with only two attacking stones on the board |

The vital points were placed away from the exact board center where possible so
these probes do not simply reward the model's center preference. The
second-line probe was corrected after inspection: it now has only two attacking
stones and measures whether the model chooses the third-line atari point, not
whether it can fill the last liberty and capture immediately.

Settings:

| Field | Value |
| --- | --- |
| Checkpoint line | `score_komi=2.5`, no-komi-input `4x64`, 200-sim continuation |
| Tested checkpoints | `100, 150, 200, 250, 300, 350, 400, 410` |
| Probe simulations | `100` |
| Probe type | All positive target tests; higher top1/top3 is better |

Summary:

| Cycle | Life/death top1/top3 | Double-atari top1/top3 | Second-line third-line-atari top1/top3 |
| ---: | ---: | ---: | ---: |
| `100` | `1/4 of 4` | `2/2 of 2` | `1/1 of 2` |
| `150` | `1/2 of 4` | `2/2 of 2` | `0/1 of 2` |
| `200` | `0/1 of 4` | `2/2 of 2` | `1/1 of 2` |
| `250` | `0/1 of 4` | `2/2 of 2` | `2/2 of 2` |
| `300` | `0/0 of 4` | `2/2 of 2` | `2/2 of 2` |
| `350` | `0/0 of 4` | `1/2 of 2` | `2/2 of 2` |
| `400` | `0/0 of 4` | `2/2 of 2` | `2/2 of 2` |
| `410` | `0/0 of 4` | `1/2 of 2` | `2/2 of 2` |

Cycle `410` details:

| Case | Target | Top1 | Target rank |
| --- | --- | --- | --- |
| `black_kill_straight_three` | `D6` | `G4` | not top10 |
| `white_kill_straight_three` | `D6` | `pass` | not top10 |
| `black_kill_bent_three` | `D6` | `G3` | not top10 |
| `white_kill_bent_three` | `D6` | `pass` | not top10 |
| `black_double_atari` | `E6` | `D5` | `3` |
| `white_double_atari` | `E6` | `E6` | `1` |
| `black_second_line_third_line_atari` | `E3` | `E3` | `1` |
| `white_second_line_third_line_atari` | `E3` | `E3` | `1` |

Observed interpretation:

- Double atari is already mostly learned in this probe design.
- Basic straight/bent-three life-death is not learned. The late checkpoints do
  worse than the early ones on these fixed vital-point tests, and White-to-play
  life/death often prefers `pass`.
- Second-line third-line atari is learned in this probe. After correcting the
  target from `E1` to `E3`, cycle `410` has both colors choosing the target as
  top1.
- These are deliberately small probes. They are useful as diagnostics for
  tactical understanding, but should not be treated as a complete tsumego
  benchmark.

## Score-Komi 4.5 Continuation Configuration

Reason for change:

- The `score_komi=2.5` 200-simulation continuation remained strongly
  Black-skewed.
- At stop time, the latest available cycle was `410`.

Stopped source run:

- Directory:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-200sims-noise-aug-16h-20260606`
- Latest checkpoint:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-200sims-noise-aug-16h-20260606/latest.pt`
- The old finalizer for this `score_komi=2.5` run was stopped before it could
  consume GPU time on the superseded eval.

Stop snapshot:

| Field | Value |
| --- | ---: |
| Cycle | `410` |
| Total positions | `1,314,430` |
| Total train steps | `26,240` |
| Black wins / White wins | `29 / 3` |
| Black win rate | `90.62%` |
| Mean moves | `94.781` |
| First-pass median | `65.0` |
| Early first-pass `<=40` | `6/32` |
| Early-pass alert | `false` |

New run directory:

- `/root/diamondgo/artifacts/multiworker-9x9-fresh-nokomi-4x64-score4p5-200sims-noise-aug-6h-20260606`

Configuration:

| Field | Value |
| --- | --- |
| Fresh start | `false` |
| Resume source | previous `score_komi=2.5` 200-sim latest checkpoint at cycle `410` |
| Resume semantics | model, optimizer, cycle, position, and train-step counters resume; replay buffer is rebuilt |
| Input komi | `false` |
| Input planes | `3`: own stones, opponent stones, side-to-play |
| Komi metadata | `0.5` |
| Score komi | `4.5` |
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
| Time limit | `360` minutes (`6` hours) |

Expected interpretation:

- This is primarily a color-balance/terminal-label experiment. The network
  architecture and self-play search are unchanged from the previous 200-sim
  line.
- Because replay is rebuilt after the resume boundary, the model weights start
  from cycle `410`, but training examples come from new `score_komi=4.5`
  self-play.

## Dual-GPU 2x96 Score4.5 Recorded Self-Play, Cycles 5-160

Source run:

- Server directory:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-dualgpu-2x96-score4p5-margin0p2-250sims-max150-18h-20260607`
- Local recorded catalog:
  `artifacts/selfplay-recorded-dualgpu-2x96-score4p5-20260607`
- Local catalog index:
  `artifacts/selfplay-recorded-dualgpu-2x96-score4p5-20260607/index.html`
- Shared self-play viewer example:
  `http://127.0.0.1:8765/viewers/selfplay-catalog-viewer.html?dataset=selfplay-recorded-dualgpu-2x96-score4p5-20260607&cycle=160&game=20`

Configuration:

| Field | Value |
| --- | --- |
| Fresh start | `true` |
| Input komi | `false` |
| Input planes | `3`: own stones, opponent stones, side-to-play |
| Score komi | `4.5` |
| Score margin reward scale | `0.2` |
| Channels / residual blocks | `96 / 2` |
| MCTS simulations | `250` for training self-play |
| Workers | `12` |
| Games per worker / cycle | `8 / 96` |
| Self-play devices | `cuda:0,cuda:1` |
| Max moves | `150` |
| Root Dirichlet noise | alpha `0.15`, fraction `0.25` |
| Root policy temperature | `1.1` |
| Move temperature | `1.0` through move `16`, then `0.25` |
| Checkpoint interval | every `5` cycles through cycle `50`, then every `10` cycles |
| Recorded self-play interval | every `5` cycles |

Recorded-cycle summary:

| Cycle | Games | Positions | Black wins | White wins | Black win rate | Mean moves | Early pass `<40` | Capture fraction | Mean Black margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5` | `96` | `9512` | `52` | `44` | `54.17%` | `99.083` | `31.25%` | `0.1340` | `+3.802` |
| `25` | `96` | `10877` | `54` | `42` | `56.25%` | `113.302` | `13.54%` | `0.2102` | `+1.240` |
| `50` | `96` | `10403` | `68` | `28` | `70.83%` | `108.365` | `8.33%` | `0.1728` | `+2.312` |
| `80` | `96` | `10976` | `65` | `31` | `67.71%` | `114.333` | `5.21%` | `0.1535` | `+2.615` |
| `110` | `96` | `9996` | `61` | `35` | `63.54%` | `104.125` | `8.33%` | `0.1414` | `+0.906` |
| `140` | `96` | `9485` | `63` | `33` | `65.62%` | `98.802` | `7.29%` | `0.1264` | `+0.719` |
| `160` | `96` | `10593` | `72` | `24` | `75.00%` | `110.344` | `9.38%` | `0.1560` | `+2.010` |

Observed interpretation:

- The run is still Black-favored despite `score_komi=4.5`, especially from
  cycle `50` onward.
- Early pass is much more common in the first recorded cycle than later. This
  suggests the early-pass pathology is still worth monitoring, but it is not
  dominating the sampled later self-play in the same way as the earlier failed
  eval slices.
- The catalog viewer should be the default display path for this run because
  it reuses recorded training games instead of creating new showcase games.

## Cross-Run Matches, Dual-GPU 2x96 Score4.5 vs Old 4x64 Score2.5

Reusable script:

- `scripts/run_cross_run_matches.py`
- Commit introducing the script: `8850431 Add reusable cross-run match script`

The script supports explicit checkpoint pairs and close/approximate pair lists.
It does not require hardcoded same-cycle comparisons.

Match settings:

| Field | Value |
| --- | --- |
| Candidate line | fresh dual-GPU `2x96`, `score_komi=4.5` |
| Opponent line | earlier fresh no-komi-input `4x64`, `score_komi=2.5` |
| Games per pair | `10` |
| Color split | candidate Black `5`, candidate White `5` |
| Candidate simulations | `100` |
| Opponent simulations | `100` |
| Match rules source | candidate config |
| Match score komi | `4.5` |
| Match max moves | `150` |

Artifacts:

- Exact/available pair dashboard:
  `artifacts/crossrun-exact-and-x2-20260607/dashboard.html`
- Exact/available pair games:
  `artifacts/crossrun-exact-and-x2-20260607/games_dashboard.html`
- Close-pair dashboard:
  `artifacts/crossrun-close-20260607/dashboard.html`
- Close-pair games:
  `artifacts/crossrun-close-20260607/games_dashboard.html`

Availability note:

- The old server was no longer reachable, so exact same-cycle and exact
  double-cycle tests were limited by locally/server-side archived checkpoint
  inventory.
- Exact available same-cycle: old cycle `80`.
- Exact available double-cycle: new cycle `50` versus old cycle `100`.
- Additional close-pair tests use nearby old checkpoints where exact
  counterparts were unavailable.

Exact/available pairs:

| Pair | Candidate wins | Candidate as Black | Candidate as White | Early pass `<40` |
| --- | ---: | ---: | ---: | ---: |
| `new80-vs-old80` | `8/10` | `5/5` | `3/5` | `20.00%` |
| `new50-vs-old100` | `5/10` | `2/5` | `3/5` | `0.00%` |

Close same-ish pairs:

| Pair | Candidate wins | Candidate as Black | Candidate as White | Early pass `<40` |
| --- | ---: | ---: | ---: | ---: |
| `new50-vs-old60` | `7/10` | `3/5` | `4/5` | `10.00%` |
| `new110-vs-old100` | `6/10` | `3/5` | `3/5` | `10.00%` |
| `new140-vs-old150` | `5/10` | `3/5` | `2/5` | `10.00%` |
| `new160-vs-old150` | `8/10` | `4/5` | `4/5` | `30.00%` |

Close double-ish pairs:

| Pair | Candidate wins | Candidate as Black | Candidate as White | Early pass `<40` |
| --- | ---: | ---: | ---: | ---: |
| `new80-vs-old150` | `4/10` | `4/5` | `0/5` | `0.00%` |
| `new110-vs-old200` | `1/10` | `1/5` | `0/5` | `10.00%` |
| `new140-vs-old300` | `3/10` | `1/5` | `2/5` | `10.00%` |
| `new160-vs-old300` | `3/10` | `3/5` | `0/5` | `10.00%` |

Observed interpretation:

- At similar early checkpoint counts, the fresh `2x96`, `score_komi=4.5` line
  looks competitive with or ahead of the old `4x64`, `score_komi=2.5` line.
- Against roughly double-age old checkpoints, the new line is clearly not
  consistently ahead yet.
- The color split is still informative. Several weak results are especially
  weak when the new model is White, so future reports should keep color-split
  win rates rather than only aggregate wins.
- Each pair has only `10` games, so these are quick diagnostic matches rather
  than final strength estimates.

## Preflight Cross-Run Matches, Dual-GPU 2x96 Score4.5 vs Late Old 4x64 Score5.5

Context:

- This was the first cross-run comparison run before the user clarified that
  the desired first report was same-cycle or close-cycle comparison.
- It is retained as a preflight record because it measures the new early
  `2x96`, `score_komi=4.5` line against a much later old `4x64`,
  `score_komi=5.5` continuation.
- It should not be mixed with the same-ish or double-ish `score_komi=2.5`
  comparison above.

Artifacts:

- Server directory:
  `/root/diamondgo/artifacts/crossrun-dualgpu2x96-score4p5-vs-old4x64-score5p5-20260607`
- Local dashboard:
  `artifacts/crossrun-dualgpu2x96-score4p5-vs-old4x64-score5p5-20260607/dashboard.html`
- Local games dashboard:
  `artifacts/crossrun-dualgpu2x96-score4p5-vs-old4x64-score5p5-20260607/games_dashboard.html`
- Local summary:
  `artifacts/crossrun-dualgpu2x96-score4p5-vs-old4x64-score5p5-20260607/summary.json`

Match settings:

| Field | Value |
| --- | --- |
| Candidate line | fresh dual-GPU `2x96`, `score_komi=4.5` |
| Opponent line | late continuation `4x64`, `score_komi=5.5` |
| Games per pair | `10` |
| Color split | candidate Black `5`, candidate White `5` |
| Candidate simulations | `100` |
| Opponent simulations | `100` |
| Match rules source | candidate config |
| Match score komi | `4.5` |
| Match max moves | `150` |

Preflight pairs:

| Pair | Candidate wins | Candidate as Black | Candidate as White | Early pass `<40` |
| --- | ---: | ---: | ---: | ---: |
| `new050-vs-old530` | `0/10` | `0/5` | `0/5` | `30.00%` |
| `new080-vs-old560` | `3/10` | `1/5` | `2/5` | `20.00%` |
| `new110-vs-old590` | `1/10` | `0/5` | `1/5` | `30.00%` |
| `new140-vs-old620` | `3/10` | `1/5` | `2/5` | `10.00%` |
| `new160-vs-old650` | `1/10` | `0/5` | `1/5` | `0.00%` |

Observed interpretation:

- The new early `2x96` run is far behind the late old `4x64`,
  `score_komi=5.5` continuation in this preflight matchup.
- Because the opponent checkpoints are hundreds of cycles later, this is mostly
  a "late old line remains much stronger" check, not a fair architecture or
  same-training-age comparison.

## Dual-GPU 2x96 Score4.5 18h + 3h Continuation

Context:

- The 18-hour fresh dual-GPU `2x96`, `score_komi=4.5`, `250`-simulation run
  was allowed to finish, then a queue watcher immediately launched a 3-hour
  continuation from the 18-hour run's `latest.pt`.
- The original 18-hour finalizer was intentionally stopped before it could run,
  so eval would not compete with the requested continuation training. Eval and
  tactical probes were run after the +3h continuation instead.
- The continuation preserves checkpoint cycle numbering from the resumed
  checkpoint. The run ended at cycle `272`, not at a fresh cycle `36`.

Artifacts:

- Local archive:
  `artifacts/server-runs/20260607-dualgpu-2x96-18h-plus3h/diamondgo-dualgpu-2x96-18h-plus3h-20260607.tar.gz`
- Local extracted root:
  `artifacts/server-runs/20260607-dualgpu-2x96-18h-plus3h/artifacts/`
- 18h training directory:
  `artifacts/server-runs/20260607-dualgpu-2x96-18h-plus3h/artifacts/multiworker-9x9-fresh-dualgpu-2x96-score4p5-margin0p2-250sims-max150-18h-20260607`
- +3h continuation directory:
  `artifacts/server-runs/20260607-dualgpu-2x96-18h-plus3h/artifacts/multiworker-9x9-cont-dualgpu-2x96-score4p5-margin0p2-250sims-max150-3h-after18h-20260607`
- Continuation eval suite:
  `artifacts/server-runs/20260607-dualgpu-2x96-18h-plus3h/artifacts/eval-suite-cont-dualgpu-2x96-score4p5-margin0p2-250sims-train-100sims-eval-3h-after18h-20260607`
- Continuation tactical probes:
  `artifacts/server-runs/20260607-dualgpu-2x96-18h-plus3h/artifacts/tactical-cont-dualgpu-2x96-score4p5-margin0p2-250sims-train-100sims-eval-3h-after18h-20260607`

Training settings:

| Field | Value |
| --- | --- |
| Architecture | `2` residual blocks x `96` channels |
| Parameters | `356,957` |
| Input komi | `false` |
| Komi metadata | `0.5` |
| Score komi | `4.5` |
| Terminal dead-stone cleanup | `false` |
| Score margin reward scale | `0.2` |
| Rules backend | `sgfmill` |
| Self-play simulations | `250` |
| Workers | `12` |
| Self-play devices | `cuda:0,cuda:1` |
| Trainer device | `cuda:0` |
| Games per worker | `8` |
| Games per cycle | `96` |
| Max moves | `150` |
| Train steps per cycle | `64` |
| Batch size | `256` |
| Replay size | `100,000` |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| `c_puct` | `1.5` |
| Root noise | Dirichlet alpha `0.15`, fraction `0.25` |
| Root policy temperature | `1.1` |
| Move temperature | `1.0` for first `16` moves, then `0.25` |
| Augmentation | Random dihedral transforms |
| SGF/trace archive | Every `5` cycles |

Training results:

| Segment | Start time | End time | Final cycle | Segment cycles | Total positions at end | Recent speed | Checkpoints | SGF/trace records |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 18h fresh run | `2026-06-06 23:51 +08` | `2026-06-07 17:52 +08` | `236` | `236` | `2,452,260` | `38.53 pos/s` over cycles `232-236` | `28` | `47/47` |
| +3h continuation | `2026-06-07 17:52 +08` | `2026-06-07 20:57 +08` | `272` | `36` | `2,857,846` | `36.82 pos/s` over cycles `268-272` | `4` | `7/7` |

Final self-play snapshot:

| Segment | Cycle | Games | Positions | Black win rate | White win rate | Early pass `<40` | Mean moves | Max-move games | Color alert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 18h final | `236` | `96` | `11,252` | `64.58%` | `35.42%` | `16.67%` | `117.208` | `32` | none |
| +3h final | `272` | `96` | `11,026` | `76.04%` | `23.96%` | `12.50%` | `114.854` | `33` | `black` |
| +3h recent 5 | `268-272` | `480` | n/a | `75.83%` | `24.17%` | `15.62%` | n/a | n/a | n/a |

GPU monitor during the +3h continuation:

| GPU | Mean util | Recent mean util | Max util | Max memory |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `38.21%` | `37.62%` | `90%` | `4017 MiB` |
| `1` | `36.11%` | `35.37%` | `87%` | `1991 MiB` |

Continuation eval settings:

| Field | Value |
| --- | --- |
| Eval simulations | `100` |
| Games per match | `20` |
| Max moves | `150` |
| Sample SGFs per match | `2` |
| Opponents requested | `initial,previous` |
| Step tiers requested | `50,200,500` |

Important eval caveat:

- `eval_suite` chooses "previous" from the checkpoints available inside the
  checkpoint directory being evaluated. In this continuation directory,
  checkpoint files are `cycle-00240`, `cycle-00250`, `cycle-00260`,
  `cycle-00270`, plus `latest` cycle `272`.
- Therefore, the useful n+50-style continuation comparison is
  `cycle-00272` versus `cycle-00250` in the `step-00050-vs-previous` report.
- The `step-00200-vs-previous` and `step-00500-vs-previous` reports include
  only `latest`, so their "previous" opponent is effectively still the initial
  model. Do not read those two as true previous-checkpoint comparisons.

Eval results:

| Eval report | Candidate | Opponent | Win rate | Wins | Black wins | White wins | First pass median | First pass `<40` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `step-00050-vs-initial` | `cycle-00250` | `cycle-00000` | `55.0%` | `11/20` | `1/10` | `10/10` | `30.5` | `17/20` |
| `step-00050-vs-initial` | `cycle-00272` | `cycle-00000` | `85.0%` | `17/20` | `7/10` | `10/10` | `30.0` | `15/20` |
| `step-00050-vs-previous` | `cycle-00250` | `cycle-00000` | `55.0%` | `11/20` | `1/10` | `10/10` | `30.5` | `17/20` |
| `step-00050-vs-previous` | `cycle-00272` | `cycle-00250` | `85.0%` | `17/20` | `10/10` | `7/10` | `57.0` | `1/20` |
| `step-00200-vs-initial` | `cycle-00272` | `cycle-00000` | `75.0%` | `15/20` | `5/10` | `10/10` | `33.0` | `15/20` |
| `step-00500-vs-initial` | `cycle-00272` | `cycle-00000` | `75.0%` | `15/20` | `5/10` | `10/10` | `33.0` | `15/20` |

Tactical probe results for continuation latest:

| Case | Target | Top1 | Top3 | Rank |
| --- | --- | --- | --- | ---: |
| `black_capture_one_stone` | `C8` | false | false | n/a |
| `white_capture_one_stone` | `F5` | true | true | `1` |
| `black_escape_atari` | `D7` | false | false | `5` |
| `white_escape_atari` | `F3` | false | false | n/a |

Observed interpretation:

- The +3h continuation did continue learning by the eval metric:
  `cycle-272` beat `cycle-250` by `17/20` in the valid `step-50 vs previous`
  match.
- The color issue is still substantial. Final self-play is about
  `76%` Black wins, and eval versus initial often has the candidate winning all
  White games. This should be read together with the pass metrics, not as a
  clean strength number.
- The `cycle-272 vs cycle-250` eval has much healthier pass timing than the
  initial-opponent eval: first-pass `<=40` is `1/20`, while vs initial remains
  `15/20`.
- Tactical probes are still weak. Only `white_capture_one_stone` is top1/top3;
  the black capture and both atari-escape probes fail. This supports the user's
  earlier concern that higher aggregate win rate does not necessarily mean the
  model has learned the local tactics we care about.
