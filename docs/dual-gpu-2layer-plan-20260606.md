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
