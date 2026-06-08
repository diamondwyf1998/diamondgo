set -euo pipefail

cd /root/diamondgo

BASE_OUT="artifacts/multiworker-9x9-cont-dualgpu-2x96-score6p5-margin0p2-400sims-max150-7d-after18hplus3h-20260607"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$BASE_OUT/latest.pt}"
TRAIN_OUT="artifacts/multiworker-9x9-cont-dualgpu-2x96-score6p5-cleanup-margin0p2-400sims-max150-7d-after18hplus3h-20260607"
LOG="$TRAIN_OUT/train.log"
MONITOR="$TRAIN_OUT/gpu_monitor.csv"
STATUS_LOG="$TRAIN_OUT/status_3h.log"

if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "missing resume checkpoint: $RESUME_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$TRAIN_OUT"
cat > "$TRAIN_OUT/run_notes.md" <<NOTES
# Continuation Dual-GPU 2x96, Score Komi 6.5, Cleanup, Margin Reward 0.2, 400 Sims, 7 Days

- created_time: $(date --iso-8601=seconds)
- source_commit_on_server: $(git rev-parse --short HEAD 2>/dev/null || cat SOURCE_COMMIT 2>/dev/null || echo unknown)
- purpose: continue the score6.5/400-sim run with conservative terminal dead-stone cleanup enabled
- fresh_start: false
- resume_checkpoint: $RESUME_CHECKPOINT
- base_run: $BASE_OUT
- comparison_warning: this continuation preserves checkpoint cycle numbering from the resumed checkpoint; compare by total positions/games/wall-clock/eval, not raw cycle ID alone
- cleanup_transition: the first score6.5/400-sim segment ran a few cycles with terminal cleanup disabled, then this cleanup-enabled continuation starts from its latest checkpoint
- model_input_komi: false
- input_planes: 3
- komi_metadata: 0.5
- score_komi: 6.5
- terminal_dead_stone_cleanup: true
- cleanup_scope: remove only conservatively detected obvious dead groups; edge deaths, seki, ko, and complex life-and-death are not solved
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
- selfplay_simulations: 400
- train_steps_per_cycle: 64
- batch_size: 256
- replay_size: 100000
- optimizer: AdamW, learning rate 0.001, weight decay 0.0001
- c_puct: 1.5
- root_noise: Dirichlet alpha 0.15, fraction 0.25
- root_policy_temperature: 1.1
- move_temperature: 1.0 for first 16 moves, then 0.25
- augmentation: random dihedral board symmetries during training
- checkpoint_every: 10 cycles after the early dense window
- early_checkpoint_cycles: 50
- early_checkpoint_every: 5
- complete_sgf_trace_archive: every 5 cycles in cycle-records/
- record_every: 5
- time_limit_minutes: 10080
- server_status_monitor: appends a text snapshot to status_3h.log every 3 hours
- old scripts/checkpoints: preserved; this continuation uses a new output directory and the dual-GPU entrypoint
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

(
  while true; do
    {
      echo "status_time=$(date --iso-8601=seconds)"
      echo "run_script_pid=$$"
      echo "latest_metric=$(tail -n 1 "$TRAIN_OUT/metrics.jsonl" 2>/dev/null || true)"
      echo "latest_checkpoints=$(find "$TRAIN_OUT/checkpoints" -maxdepth 1 -name 'cycle-*.pt' 2>/dev/null | sort | tail -5 | tr '\n' ' ')"
      echo "latest_cycle_records=$(find "$TRAIN_OUT/cycle-records" -maxdepth 1 -name 'cycle-*.sgf' 2>/dev/null | sort | tail -5 | tr '\n' ' ')"
      echo "gpu_snapshot=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')"
      echo "disk_snapshot=$(df -h /root /root/autodl-tmp 2>/dev/null | tr '\n' ';')"
      echo "---"
    } >> "$STATUS_LOG"
    sleep 10800
  done
) &
STATUS_PID=$!

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  kill "$STATUS_PID" 2>/dev/null || true
}
trap cleanup EXIT

PYTHONPATH=src /root/miniconda3/bin/python -u -m diamondgo.multiworker_train_dualgpu \
  --json \
  --out-dir "$TRAIN_OUT" \
  --resume "$RESUME_CHECKPOINT" \
  --device cuda:0 \
  --selfplay-devices cuda:0,cuda:1 \
  --rules sgfmill \
  --komi 0.5 \
  --score-komi 6.5 \
  --no-input-komi \
  --terminal-dead-stone-cleanup \
  --score-margin-reward-scale 0.2 \
  --time-limit-minutes 10080 \
  --cycles 100000 \
  --workers 12 \
  --games-per-worker 8 \
  --max-moves 150 \
  --simulations 400 \
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
  --early-checkpoint-cycles 50 \
  --early-checkpoint-every 5 \
  --record-every 5 \
  > "$LOG" 2>&1
