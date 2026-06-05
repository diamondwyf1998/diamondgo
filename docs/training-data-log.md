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
