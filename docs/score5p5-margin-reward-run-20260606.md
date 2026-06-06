# Score 5.5 margin reward continuation

Date: 2026-06-06

## Intent

White win rate looked high in the score-komi 6.5 continuation. This run lowers scoring komi to 5.5 to reduce White's scoring advantage, while testing the bounded score-margin value target.

This is a continuation run, not a fresh start. It should resume from the latest checkpoint of the score-komi 6.5 / 200-sim run, and write to a new output directory so earlier checkpoints remain available.

## Planned Config

- `rules_backend`: `sgfmill`
- `komi`: `0.5`
- `score_komi`: `5.5`
- `input_komi`: `false`
- `terminal_dead_stone_cleanup`: `false`
- `score_margin_reward_scale`: `0.2`
- `simulations`: `300`
- `max_moves`: `150`
- `channels`: `64`
- `residual_blocks`: `4`
- `workers`: `8`
- `games_per_worker`: `4`
- `train_steps_per_cycle`: `64`
- `batch_size`: `256`
- `replay_size`: `100000`
- `learning_rate`: `0.001`
- `weight_decay`: `0.0001`
- `c_puct`: `1.5`
- `temperature`: `1.0`
- `temperature_moves`: `16`
- `late_temperature`: `0.25`
- `root_dirichlet_alpha`: `0.15`
- `root_noise_fraction`: `0.25`
- `root_policy_temperature`: `1.1`
- `augment_dihedral`: `true`
- `checkpoint_every`: `10`
- `time_limit_minutes`: `360`

## Notes

- Score-margin value target is bounded by design:
  `sign(score_margin) * (2/5 + min(abs(score_margin) ** 0.25 / 5 * scale, 3/5))`.
- With `scale=0.2`, the margin term is intentionally small for this first run.
- Terminal dead-stone cleanup remains off for safety, so any result change can be attributed mainly to scoring komi, MCTS count, max-move limit, and the bounded margin target.
- Server launch is queued behind any existing score-komi 6.5 finalizer/eval or `pairwise_50cycle_extra6` process. Do not kill those eval jobs; the 5.5 run should start only after they finish.

## Completed Result

Training finished on 2026-06-06. The source commit recorded for the queued run was
`24ef8f2`. The run continued from the previous score-komi line and wrote:

- Server train dir:
  `/root/diamondgo/artifacts/multiworker-9x9-cont-nokomi-4x64-score5p5-margin0p2-300sims-max150-6h-20260606`
- Server eval dir:
  `/root/diamondgo/artifacts/eval-suite-cont-nokomi-4x64-score5p5-margin0p2-300sims-train-100sims-eval-6h-20260606`
- Server tactical dir:
  `/root/diamondgo/artifacts/tactical-cont-nokomi-4x64-score5p5-margin0p2-300sims-train-100sims-eval-6h-20260606`
- Local artifact copy:
  `artifacts/server-runs/20260606-score5p5-margin0p2`

Raw artifacts are kept locally under `artifacts/`, which is git-ignored. The
GitHub repo keeps the source, scripts, notes, and compact result record rather
than committing generated checkpoints and dashboards into repository history.

Final training metrics:

| item | value |
|---|---:|
| completed cycles | `523` to `666` |
| total positions | `1,998,015` |
| total train steps | `42,624` |
| final cycle seconds | `177.619` |
| final positions in cycle | `2,456` |
| final loss / policy / value | `1.122594 / 1.029778 / 0.092816` |
| value target range | `[-0.4916, 0.4916]` |
| final self-play B/W wins | `25 / 7` |
| final self-play B/W win rate | `78.12% / 21.88%` |
| final self-play color alert | `black` |
| final ended by pass / max moves | `28 / 4` |
| final mean moves | `76.75` |
| final early first-pass <=40 | `18.75%` |
| final mean black score margin | `+7.156` |
| final capture moves / stones | `342 / 741` |

Eval summary:

| tier | candidate | opponent | win rate | B wins | W wins | first-pass median | first pass <=40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| every 50 vs initial | `550` | `0` | `95%` | `9/10` | `10/10` | `21.5` | `18/20` |
| every 50 vs initial | `600` | `0` | `95%` | `9/10` | `10/10` | `26.5` | `20/20` |
| every 50 vs initial | `650` | `0` | `90%` | `8/10` | `10/10` | `25.5` | `17/20` |
| every 50 vs initial | `666` | `0` | `95%` | `9/10` | `10/10` | `25.0` | `16/20` |
| every 50 vs previous | `600` | `550` | `80%` | `7/10` | `9/10` | `59.5` | `2/20` |
| every 50 vs previous | `650` | `600` | `70%` | `8/10` | `6/10` | `64.5` | `0/20` |
| every 50 vs previous | `666` | `650` | `55%` | `8/10` | `3/10` | `64.5` | `0/20` |
| every 200 vs initial | `600` | `0` | `65%` | `4/10` | `9/10` | `25.0` | `18/20` |
| every 200 vs initial | `666` | `0` | `100%` | `10/10` | `10/10` | `28.0` | `19/20` |
| every 200 vs previous | `666` | `600` | `55%` | `7/10` | `4/10` | `56.0` | `0/20` |
| every 500 vs initial | `666` | `0` | `90%` | `8/10` | `10/10` | `26.5` | `19/20` |

The eval still shows strong early-pass pathology against the initial opponent,
so the high win rates should not be read as clean strength gains. The
checkpoint-vs-previous matches are more useful here: by cycle 666, the gain over
cycle 650/600 is only about `55%` in the sampled matches.

Tactical probe summary:

| case | target | top1 | top3 | rank |
|---|---:|---:|---:|---:|
| black_capture_one_stone | `C8` | false | false | `10` |
| white_capture_one_stone | `F5` | true | true | `1` |
| black_escape_atari | `D7` | true | true | `1` |
| white_escape_atari | `F3` | false | true | `2` |

## Follow-Up Logging Change

After this run, the training scripts were updated so future runs archive
complete cycle records every ten cycles:

- `latest-cycle.sgf` and `latest-cycle-trace.json` remain the rolling latest
  files.
- On cycles `10, 20, 30, ...`, both `multiworker_train.py` and
  `overnight_train.py` also write `cycle-records/cycle-xxxxx.sgf` and
  `cycle-records/cycle-xxxxx-trace.json`.
- Multi-game SGF exports are now written as a proper SGF collection, one game
  tree per self-play game, so the files are easier to inspect in Sabaki.

This change was not present during the score-5.5 run above, so that run only has
the final rolling `latest-cycle.sgf`/trace pair.
