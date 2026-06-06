set -euo pipefail

cd /root/diamondgo

TRAIN_OUT="artifacts/multiworker-9x9-fresh-dualgpu-2x96-score5p5-margin0p2-300sims-max150-6h-20260606"
LOG="$TRAIN_OUT/train.log"
MONITOR="$TRAIN_OUT/gpu_monitor.csv"

mkdir -p "$TRAIN_OUT"
cat > "$TRAIN_OUT/run_notes.md" <<NOTES
# Fresh Dual-GPU 2x96, Score Komi 5.5, Margin Reward 0.2

- created_time: $(date --iso-8601=seconds)
- source_commit: $(git rev-parse --short HEAD 2>/dev/null || cat SOURCE_COMMIT 2>/dev/null || echo unknown)
- fresh_start: true
- model_input_komi: false
- input_planes: 3
- komi_metadata: 0.5
- score_komi: 5.5
- terminal_dead_stone_cleanup: false
- score_margin_reward_scale: 0.2
- value_target_formula_when_margin_reward_enabled: sign(score_margin) * (2/5 + min(abs(score_margin) ** 0.25 / 5 * scale, 3/5))
- channels: 96
- residual_blocks: 2
- trainable_params: 356957
- trainer_device: cuda:0
- selfplay_devices: cuda:0,cuda:1
- workers: 12
- games_per_worker: 8
- games_per_cycle: 96
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
- complete_sgf_trace_archive: every 10 cycles in cycle-records/
- time_limit_minutes: 360
- old scripts/checkpoints: preserved; this run uses a new output directory and the new multiworker_train_dualgpu entrypoint
NOTES

(
  echo "timestamp,gpu_index,gpu_name,gpu_util_pct,mem_util_pct,mem_used_mib,mem_total_mib,power_w"
  while true; do
    ts=$(date --iso-8601=seconds)
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits |
      while IFS= read -r stats; do
        echo "$ts,$stats"
      done
    sleep 60
  done
) > "$MONITOR" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT

PYTHONPATH=src /root/miniconda3/bin/python -u -m diamondgo.multiworker_train_dualgpu \
  --json \
  --out-dir "$TRAIN_OUT" \
  --device cuda:0 \
  --selfplay-devices cuda:0,cuda:1 \
  --rules sgfmill \
  --komi 0.5 \
  --score-komi 5.5 \
  --no-input-komi \
  --no-terminal-dead-stone-cleanup \
  --score-margin-reward-scale 0.2 \
  --time-limit-minutes 360 \
  --cycles 100000 \
  --workers 12 \
  --games-per-worker 8 \
  --max-moves 150 \
  --simulations 300 \
  --train-steps-per-cycle 64 \
  --batch-size 256 \
  --replay-size 100000 \
  --channels 96 \
  --residual-blocks 2 \
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
