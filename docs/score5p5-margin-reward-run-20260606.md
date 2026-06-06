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
