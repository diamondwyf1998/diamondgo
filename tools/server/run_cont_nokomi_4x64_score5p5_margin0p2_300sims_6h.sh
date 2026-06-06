set -euo pipefail

cd /root/diamondgo

TRAIN_OUT="artifacts/multiworker-9x9-cont-nokomi-4x64-score5p5-margin0p2-300sims-max150-6h-20260606"
RESUME="artifacts/multiworker-9x9-fresh-nokomi-4x64-score6p5-200sims-noise-aug-5h-20260606/latest.pt"
LOG="$TRAIN_OUT/train.log"
MONITOR="$TRAIN_OUT/gpu_monitor.csv"

mkdir -p "$TRAIN_OUT"
cat > "$TRAIN_OUT/run_notes.md" <<NOTES
# No-Komi-Input 4x64, Score Komi 5.5, Margin Reward 0.2 Continuation

- created_time: $(date --iso-8601=seconds)
- source_commit: 1436c45
- fresh_start: false
- resume_checkpoint: $RESUME
- resume_semantics: model weights, optimizer state, cycle count, position count, and train step count continue from the resume checkpoint; replay buffer is rebuilt in this new run directory
- reason: score_komi=6.5 latest self-play showed White win rate above 80%; lower scoring komi to 5.5 while adding a small bounded score-margin target
- model_input_komi: false
- input_planes: 3
- komi_metadata: 0.5
- score_komi: 5.5
- terminal_dead_stone_cleanup: false
- score_margin_reward_scale: 0.2
- value_target_formula_when_margin_reward_enabled: sign(score_margin) * (2/5 + min(abs(score_margin) ** 0.25 / 5 * scale, 3/5))
- channels: 64
- residual_blocks: 4
- trainable_params: 316669
- workers: 8
- games_per_worker: 4
- games_per_cycle: 32
- max_moves: 150
- selfplay_simulations: 300
- train_steps_per_cycle: 64
- batch_size: 256
- replay_size: 100000
- optimizer: AdamW, learning rate 0.001, weight decay 0.0001
- c_puct: 1.5
- root_noise: Dirichlet alpha 0.15, fraction 0.25
- root_policy_temperature: 1.1
- move_temperature: 1.0 for first 16 moves, then 0.25
- augmentation: random dihedral board symmetries during training
- checkpoint_every: 10 cycles
- time_limit_minutes: 360
- old scripts/checkpoints: preserved; this run uses a new output directory
NOTES

(
  echo "timestamp,gpu_util_pct,mem_util_pct,mem_used_mib,mem_total_mib,power_w"
  while true; do
    ts=$(date --iso-8601=seconds)
    stats=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits)
    echo "$ts,$stats"
    sleep 60
  done
) > "$MONITOR" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT

PYTHONPATH=src /root/miniconda3/bin/python -u -m diamondgo.multiworker_train \
  --json \
  --out-dir "$TRAIN_OUT" \
  --resume "$RESUME" \
  --device cuda \
  --rules sgfmill \
  --komi 0.5 \
  --score-komi 5.5 \
  --no-input-komi \
  --no-terminal-dead-stone-cleanup \
  --score-margin-reward-scale 0.2 \
  --time-limit-minutes 360 \
  --cycles 100000 \
  --workers 8 \
  --games-per-worker 4 \
  --max-moves 150 \
  --simulations 300 \
  --train-steps-per-cycle 64 \
  --batch-size 256 \
  --replay-size 100000 \
  --channels 64 \
  --residual-blocks 4 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --c-puct 1.5 \
  --temperature 1.0 \
  --temperature-moves 16 \
  --late-temperature 0.25 \
  --root-dirichlet-alpha 0.15 \
  --root-noise-fraction 0.25 \
  --root-policy-temperature 1.1 \
  --augment-dihedral \
  --checkpoint-every 10 \
  > "$LOG" 2>&1
