# Dual-GPU 2-Layer Training Plan

Date: 2026-06-06

This is a planned run for the replacement server:

- GPU: `2 x NVIDIA GeForce RTX 3080 Ti`, `12 GiB` each
- CPU: `Intel Xeon Silver 4214R`, observed as `48` logical CPUs
- Memory: about `440 GiB`
- Source location: `/root/diamondgo`, cloned from GitHub
- Artifact location: `/root/diamondgo/artifacts`, symlinked to
  `/root/autodl-tmp/diamondgo-artifacts`

## Intent

Test how far a 2-layer model can get when using the newer training tricks rather
than the first old 2-layer scripts. The architecture returns to 2 residual
blocks, but the trunk channel count is set to `96`, interpreted as `1.5x` the
recent `64` feature channels.

This run is mainly a comparison experiment. Its cycle number should not be
compared directly with earlier runs, because the amount of self-play per cycle
changed:

- recent single-GPU 4x64 line: `8 workers x 4 games = 32 games/cycle`
- planned dual-GPU 2x96 line: `12 workers x 8 games = 96 games/cycle`

For fair comparisons, prefer total positions, self-play games, wall-clock time,
and checkpoint-vs-checkpoint eval, not raw cycle IDs.

Parameter counts for reference:

| model | parameters |
|---|---:|
| `2x32` | `54,173` |
| `2x48` | `102,221` |
| `2x64` | `168,701` |
| `2x96` | `356,957` |
| `4x64` | `316,669` |

## New Script

The old training script is preserved. The new dual-GPU entrypoint is:

- `src/diamondgo/multiworker_train_dualgpu.py`

It keeps one trainer model on `--device`, while assigning self-play workers
round-robin over `--selfplay-devices`, for example `cuda:0,cuda:1`.

## Planned Main Config

| item | value |
|---|---:|
| board | `9x9` |
| rules backend | `sgfmill` |
| model residual blocks | `2` |
| model channels | `96` |
| input komi plane | `false` |
| komi stored in model config | `0.5` |
| scoring komi | `5.5` |
| terminal dead-stone cleanup | `false` |
| score-margin reward scale | `0.2` |
| MCTS simulations | `300` |
| max moves | `150` |
| workers | `12` |
| games per worker | `8` |
| games per cycle | `96` |
| trainer device | `cuda:0` |
| self-play devices | `cuda:0,cuda:1` |
| train steps per cycle | `64` |
| train batch size | `256` |
| replay size | `100000` |
| learning rate | `0.001` |
| weight decay | `0.0001` |
| c_puct | `1.5` |
| temperature | `1.0` |
| temperature moves | `16` |
| late temperature | `0.25` |
| root Dirichlet alpha | `0.15` |
| root noise fraction | `0.25` |
| root policy temperature | `1.1` |
| dihedral augmentation | `true` |
| checkpoint every | `10` |
| complete cycle SGF/trace archive | every `10` cycles |

Before a long run, do a short smoke/throughput check with fewer cycles and watch
both GPUs separately. The target diagnostic numbers are:

- positions per hour
- per-GPU utilization and memory
- average network batch
- legal-actions seconds
- state-copy seconds
- early pass rate and color-bias alerts

## 20-Minute Throughput Check

Run date: 2026-06-07.

Output paths:

- Server:
  `/root/diamondgo/artifacts/multiworker-9x9-fresh-dualgpu-2x96-score5p5-margin0p2-300sims-max150-20m-20260607`
- Local copy:
  `artifacts/server-runs/20260607-dualgpu-2x96-20m`

This was a fresh-start throughput check with the planned dual-GPU 2x96 config.
The nominal time limit was `20` minutes, but the training loop only checks the
limit between cycles. Cycle 6 had already started near the time boundary, so the
run naturally finished after 6 full cycles.

Summary:

| item | value |
|---|---:|
| completed cycles | `6` |
| games per cycle | `96` |
| total self-play games | `576` |
| total positions | `50,479` |
| total train steps | `384` |
| average cycle seconds | `232.617` |
| average self-play throughput | `36.454 positions/s` |
| average network batch | `5.547` |
| max network batch | `8` |
| GPU0 utilization mean / median / max | `48.7% / 47.0% / 94.0%` |
| GPU1 utilization mean / median / max | `43.4% / 45.0% / 85.0%` |
| GPU0 memory mean / max | `3.89 GiB / 4.02 GiB` |
| GPU1 memory mean / max | `1.95 GiB / 1.99 GiB` |

Per-cycle summary:

| cycle | seconds | positions | pos/s | avg batch | B win rate | W win rate | first pass <=40 | max-move games | mean moves |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `251.916` | `8,197` | `32.836` | `4.890` | `37.50%` | `62.50%` | `63.54%` | `3` | `85.385` |
| `2` | `248.763` | `9,897` | `40.076` | `5.969` | `59.38%` | `40.62%` | `46.88%` | `1` | `103.094` |
| `3` | `240.465` | `8,470` | `35.477` | `5.328` | `62.50%` | `37.50%` | `33.33%` | `0` | `88.229` |
| `4` | `231.373` | `8,655` | `37.678` | `5.671` | `59.38%` | `40.62%` | `33.33%` | `1` | `90.156` |
| `5` | `202.304` | `7,366` | `36.725` | `5.805` | `64.58%` | `35.42%` | `23.96%` | `0` | `76.729` |
| `6` | `220.879` | `7,894` | `36.023` | `5.620` | `45.83%` | `54.17%` | `27.08%` | `0` | `82.229` |

Initial read:

- The dual-GPU worker split is functional: metrics record workers on both
  `cuda:0` and `cuda:1`.
- Throughput is materially better than the previous 4090D single-GPU
  4x64/300-sim line (`~36.5` positions/s here vs about `15.5` positions/s in
  the earlier 5.5-komi 300-sim run), but this is not a pure hardware comparison
  because this run uses `96 games/cycle`, a 2x96 model, and two GPUs.
- Average inference batch improved from roughly `3.1` in the previous line to
  `5.5` here, which supports the idea that larger per-worker game batches reduce
  tiny-GPU-batch overhead.
- GPU utilization is still not saturated. CPU/rules/MCTS orchestration remains
  a likely bottleneck: per cycle, summed legal-action time is still large.
- Early-pass behavior is still visible in fresh-start cycle 1, but it improved
  quickly during this short run. This should be monitored in any longer run.
