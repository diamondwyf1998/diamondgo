# Terminal reward add-ons

Date: 2026-06-06

这份 note 记录两个可选实验功能。它们默认关闭，目的是避免改变旧实验的含义。

## 功能

- `--terminal-dead-stone-cleanup`
  - 终局计分前，清除“明显死棋”。
  - 当前实现非常保守：只清除少于两只实眼、删除后形成的空区完全不触边、且该空区只被对方棋子包围的整块棋。
  - 边角死棋、打劫、双活、复杂死活暂不自动判断。这个功能是诊断/训练辅助，不是完整死活引擎。

- `--score-margin-reward-scale`
  - 在原本 `+1/-1` 胜负 value target 上，增加一项有符号目差奖励：
    `sign(score_margin) * abs(score_margin) ** 0.25 / 5 * scale`
  - `scale=1.0` 时，黑胜 16 目对应黑方 value target `1.4`，白方视角为 `-1.4`。

## 需要特别观察

- 现在模型 value head 仍然是 `tanh`，输出范围是 `[-1, 1]`。
- 如果开启目差奖励，训练 target 可能超过这个范围，因此 metrics 增加了：
  - `value_target_min`
  - `value_target_max`
- 如果 value loss 持续异常，下一步应该考虑去掉/调整 value head 的 `tanh`，或者把目差奖励压回 `[-1, 1]`。

## 已接入路径

- `demo_cpu.py`
- `batched_demo.py`
- `overnight_train.py`
- `multiworker_train.py`
- `eval_checkpoints.py`
- `tactical_eval.py`

训练和评测 checkpoint config 会记录这两个字段，便于之后复盘时区分实验条件。
