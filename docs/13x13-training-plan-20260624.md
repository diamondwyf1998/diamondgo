# 13x13 Training Plan, 2026-06-24

## 结论先行

- 当前 9x9 实验已停止，保留到 cycle `1171`，备份 checkpoint：
  `/root/autodl-tmp/diamondgo-checkpoint-backups/9x9_score6p5_margin0p2_600sims_stop_for_13x13_20260624-201836_cycle1171_latest.pt`
- 13x13 需要 fresh start。9x9 的 policy head 是 `82` 个动作，13x13 是 `170` 个动作，不能直接 resume 9x9 checkpoint。
- 已修复一类“代码已经支持但命令行没暴露”的问题：`multiworker_train.py`、`multiworker_train_dualgpu.py`、`batched_demo.py` 现在都能接收 `--board-size`。
- 13x13 smoke 和 30-worker stress 已通过。30 worker 需要先设置 `ulimit -n 65535`，否则默认 `1024` 会触发 PyTorch multiprocessing 的 `Too many open files`。
- 根据最新要求，正式默认搜索量从 `600` sims 下调到 `200` sims，并启用动态贴目阶梯。启动贴目改为 `2.5`，贴目调整阈值改为 `70%`，检测窗口改为 `5` cycles。

## 计划配置

- 棋盘：`13x13`
- 规则：`sgfmill`
- 模型：`4` residual blocks，`64` channels
- 输入：`history_moves=2`，`input_komi=false`
- 输入平面：己方棋子、对方棋子、执黑方、上一手、上上手，共 `5` planes
- 参数量：约 `367,717`
- 贴目逻辑：SGF/metadata `komi=0.5`，训练胜负判断从 `score_komi=2.5` 开始
- 动态贴目阶梯：`2.5, 4.5, 6.5, 7.5, 8.5`
- 动态贴目规则：最近最多 `5` 个 cycle 的平均黑胜率超过 `70%`，下一 cycle 升一档贴目；平均白胜率超过 `70%`，下一 cycle 降一档贴目
- 默认判断窗口：滚动最近 `5` 个 cycle；前 `5` 个 cycle 尚未凑满时，用已有 cycle 的平均值
- 终局：`terminal_dead_stone_cleanup=true`
- 价值目标：`score_margin_reward_scale=0.2`
- 自对弈搜索：先用 `200` simulations
- MCTS：`c_puct=1.5`
- 根节点噪声：Dirichlet `alpha=0.15`，fraction `0.25`
- 根 policy 温度：`root_policy_temperature=1.1`
- 落子采样：前 `30` 手 `temperature=0.75`，之后 `late_temperature=0.2`
- worker：正式默认 `30` workers，每 worker `8` games，即 `240` games/cycle
- 最大手数：`max_moves=250`
- 训练：`train_steps_per_cycle=64`，`batch_size=256`，`replay_size=100000`
- 优化器：AdamW，learning rate `0.0015`，weight decay `0.0001`
- 数据增强：dihedral augmentation enabled
- checkpoint：前 `50` cycle 每 `5` cycle 存一次，之后每 `10` cycle
- 棋谱：每 `5` cycle 记录完整 SGF/trace
- 搜索树：每 `20` cycle 只保留前 `5` 盘 full root trace；其他 trace 只保留 top `5` actions

## 调试结果

- 本地 CPU batched 13x13 smoke 通过：
  - SGF 写出 `SZ[13]`
  - pass action 为 `169`
  - 小模型训练一步正常
- 本地 CPU multiworker 13x13 smoke 通过：
  - `config.json` 正确写出 `board_size: 13`
  - 能产生 checkpoint、metrics、cycle record
- 服务器双 GPU 2-worker 13x13 smoke 通过：
  - selfplay devices 正确分到 `cuda:0,cuda:1`
  - debug 参数量 `367,717`
  - 能写 checkpoint 和 metrics
- 服务器 30-worker stress：
  - 默认 `ulimit -n=1024` 失败，错误是 `OSError: [Errno 24] Too many open files`
  - 加 `ulimit -n 65535` 后通过
  - debug 配置 `30 workers x 1 game`、`16 sims`、`max_moves=30`
  - cycle 用时约 `20.1s`
  - selfplay 约 `18.8s`
  - train 约 `0.6s`
  - positions/s 约 `48.0`

## 注意事项

- 这轮如果要比较 9x9 与 13x13，不能只按 cycle 编号比较。13x13 的单盘长度、动作空间、合法性检查成本都变了，应同时看 positions、games、wall-clock 和 eval。
- 动态贴目会改变训练标签分布；复盘时必须看每个 cycle 的 `score_komi`、`dynamic_score_komi` 和 `next_score_komi`，不要把整条线当成固定贴目实验。
- `root_dirichlet_alpha=0.15` 是为了尽量保持已有训练配置不变。按动作空间归一化时，13x13 更自然的 alpha 可能更接近 `0.07`，但那会额外引入一个变量，建议之后单独实验。
- `max_moves=250` 是折中值。13x13 棋盘有 `169` 个点；如果按 9x9 的 `150` 手比例外推会更接近 `313`，但第一轮先避免随机早期把单 cycle 拉得太长。

## 启动脚本

新脚本：

`tools/server/run_fresh_dualgpu_13x13_4x64_history2_autokomi_margin0p2_200sims.sh`

默认是不设时间上限的正式配置。调试时可用环境变量覆盖，例如：

```bash
TIME_LIMIT_MINUTES=10 SIMULATIONS=32 MAX_MOVES=60 WORKERS=30 GAMES_PER_WORKER=1 \
  bash tools/server/run_fresh_dualgpu_13x13_4x64_history2_autokomi_margin0p2_200sims.sh
```

## 2026-06-25 Update: Early-Pass Mask

- The first 13x13 auto-komi run was stopped at latest metric cycle `26`.
- Symptom: early pass dominated the games. Latest metric had `pass_move_fraction`
  about `0.3255`, first-pass median about `6`, and all `240` games ended by
  pass.
- New optional control: `min_pass_move=120`.
- Semantics: while a game has fewer than `120` played moves, pass is removed
  from MCTS legal actions. This affects root expansion, leaf expansion, policy
  targets, and final action sampling consistently.
- Safety edge case: if no board point is legal, pass remains legal so MCTS
  does not create an empty root.
- Optimizer change for the continuation: learning rate raised from `0.001` to
  `0.0015`.
- Caveat: this is a deliberate experiment-condition change. It should not be
  compared to the pre-mask cycles as if only training time changed.

## 2026-06-25 Update: Opening Temperature

- Changed the current continuation to use `temperature=0.75` for the first
  `30` moves, then `late_temperature=0.2`.
- This replaces the previous `temperature=0.7` for the first `16` moves.
- Purpose: keep more early exploration after pass is masked, while preserving a
  low late-game sampling temperature.

## 2026-06-25 Update: Final-Board Prediction Head

- User request: imitate the KataGo-style auxiliary prediction idea and add a
  head that predicts the terminal board distribution.
- Implemented target: black-perspective terminal ownership, one value per
  board point:
  - `+1`: black stone or black surrounded territory at the configured terminal
    scoring state.
  - `-1`: white stone or white surrounded territory at the configured terminal
    scoring state.
  - `0`: neutral/unsettled point under the current simple area ownership
    extraction.
- The target is generated after the configured terminal cleanup rule. This is
  not a full life-and-death solver; it mirrors the current conservative cleanup
  and area-scoring logic.
- Model change: `PolicyValueNet.forward()` now returns
  `(policy_logits, value, final_board)`.
- Loss change: training now uses
  `policy_loss + value_loss + final_board_loss_weight * final_board_loss`.
- Current default `final_board_loss_weight`: `0.25`.
- Data augmentation change: dihedral augmentation now applies the same board
  transform to `final_board_target` as to the input planes and policy board.
- Backward compatibility: old checkpoints without the new head can still be
  loaded for eval/play; the new head is randomly initialized in that case.
- Local smoke passed on Windows CPU:
  - 13x13 model output shapes: policy `170`, final board `169`.
  - ownership target and dihedral target transform checked by direct smoke.
  - command completed:
    `python -m diamondgo.batched_demo --json --board-size 13 --device cpu --rules sgfmill --games 1 --simulations 2 --max-moves 8 --min-pass-move 6 --train-steps 1 --channels 8 --residual-blocks 1 --no-input-komi --history-moves 2 --terminal-dead-stone-cleanup --score-margin-reward-scale 0.2 --final-board-loss-weight 0.25 ...`
  - observed final training metrics included `final_board_loss: 0.109844`.
- Local artifact paths:
  - `artifacts/local-smoke-13x13-final-board-head.sgf`
  - `artifacts/local-smoke-13x13-final-board-head.json`
  - `artifacts/visualizations/local-smoke-13x13-final-board-head.html`
  - `artifacts/visualizations/local-smoke-13x13-final-board-head.svg`

## 2026-06-25 Server Run: Final-Board Head, Fresh 13x13, 32 Workers

- User request: push the final-board prediction change to the server, clear
  stale training workers, and restart with `workers=32`.
- Server: `connect.westc.seetacloud.com:45955`.
- Before restart, the old no-final-board fresh 13x13 line was stopped and its
  output directory was preserved.
- Stale self-play worker cleanup: after the old main process was stopped,
  orphan CUDA worker PIDs from the old line were still holding GPU memory. They
  were killed from the `nvidia-smi --query-compute-apps=pid` list.
- Server dirty source backup before overwrite:
  `/root/autodl-tmp/diamondgo-source-backups/before_final_board_20260625-205900.diff.patch`
- GitHub commit after local commit/push:
  `e7087a2 Add final board prediction head`.
- New run output:
  `/root/diamondgo/artifacts/multiworker-13x13-fresh-dualgpu-4x64-history2-autokomi-start2p5-minpass120-lr0p0015-cleanup-margin0p2-200sims-max250-temp0p7-moves30-mid0p3-until100-late0p2-32w-20260625-210133`
- PID file:
  `/root/diamondgo/artifacts/dualgpu_13x13_finalboard_32w.train.pid`
- Actual Python train PID at launch verification: `10523`.
- Config highlights:
  - board size: `13`
  - model: `4x64`, `history_moves=2`, `input_komi=false`
  - self-play devices: `cuda:0,cuda:1`
  - trainer device: `cuda:0`
  - workers: `32`
  - games per worker: `8`
  - games per cycle: `256`
  - self-play simulations: `200`
  - max moves: `250`
  - min pass move: `120`
  - score komi start: `2.5`
  - dynamic komi ladder: `2.5,4.5,6.5,7.5,8.5`
  - terminal dead-stone cleanup: `true`
  - score margin reward scale: `0.2`
  - final board loss weight: `0.25`
  - temperature schedule: `0.7` for moves `[0,30)`, `0.3` for `[30,100)`,
    `0.2` afterwards
- Server verification before launch:
  - compile passed with `python -m compileall -q src scripts tests`
  - model forward shape smoke: policy `(2,170)`, value `(2,)`, final board
    `(2,169)`
  - targeted pytest passed: `16 passed`
- First cycle result:
  - cycle seconds: `552.081`
  - self-play seconds: `544.724`
  - train seconds: `1.687`
  - positions: `50035`
  - positions per second: `91.854`
  - latest loss: `5.530139`
  - policy loss: `5.118330`
  - value loss: `0.214607`
  - final board loss: `0.788811`
  - final board target mean: `0.0138`
  - final board target black fraction: `0.4725`
  - final board target white fraction: `0.4587`
  - black win rate: `0.4883`
  - white win rate: `0.5117`
  - terminal cleanup stones: black `206`, white `295`
- First-cycle comparison note: the previous no-final-board 30-worker line had
  cycle time about `541.6s` for `240` games; this 32-worker final-board line
  used about `552.1s` for `256` games, so throughput was slightly higher in
  positions per second despite the auxiliary head.
