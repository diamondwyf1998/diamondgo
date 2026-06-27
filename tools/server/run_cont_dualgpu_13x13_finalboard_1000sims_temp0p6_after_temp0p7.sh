#!/usr/bin/env bash
set -euo pipefail

cd /root/diamondgo
ulimit -n "${ULIMIT_NOFILE:-65535}" || true

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
BOARD_SIZE="${BOARD_SIZE:-13}"
SCORE_KOMI="${SCORE_KOMI:-8.5}"
SCORE_KOMI_LADDER="${SCORE_KOMI_LADDER:-2.5,4.5,6.5,7.5,8.5,9.5,10.5}"
SCORE_KOMI_ADJUST_WINDOW="${SCORE_KOMI_ADJUST_WINDOW:-5}"
SCORE_KOMI_ADJUST_THRESHOLD="${SCORE_KOMI_ADJUST_THRESHOLD:-0.70}"
SIMULATIONS="${SIMULATIONS:-1000}"
MAX_MOVES="${MAX_MOVES:-250}"
MIN_PASS_MOVE="${MIN_PASS_MOVE:-120}"
WORKERS="${WORKERS:-32}"
GAMES_PER_WORKER="${GAMES_PER_WORKER:-8}"
TIME_LIMIT_MINUTES="${TIME_LIMIT_MINUTES:-0}"
TRAIN_STEPS_PER_CYCLE="${TRAIN_STEPS_PER_CYCLE:-64}"
BATCH_SIZE="${BATCH_SIZE:-256}"
REPLAY_SIZE="${REPLAY_SIZE:-100000}"
LEARNING_RATE="${LEARNING_RATE:-0.0015}"
FINAL_BOARD_LOSS_WEIGHT="${FINAL_BOARD_LOSS_WEIGHT:-0.25}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TEMPERATURE_MOVES="${TEMPERATURE_MOVES:-30}"
MID_TEMPERATURE="${MID_TEMPERATURE:-0.3}"
MID_TEMPERATURE_UNTIL="${MID_TEMPERATURE_UNTIL:-100}"
LATE_TEMPERATURE="${LATE_TEMPERATURE:-0.2}"
ROOT_DIRICHLET_ALPHA="${ROOT_DIRICHLET_ALPHA:-0.15}"
ROOT_NOISE_FRACTION="${ROOT_NOISE_FRACTION:-0.25}"
ROOT_POLICY_TEMPERATURE="${ROOT_POLICY_TEMPERATURE:-1.1}"

PREVIOUS_OUT="${PREVIOUS_OUT:-artifacts/multiworker-13x13-cont-dualgpu-4x64-history2-autokomi-start8p5-ladder10p5-minpass120-lr0p0015-cleanup-finalboard-margin0p2-1000sims-max250-temp0p7-moves30-mid0p3-until100-late0p2-32w-after-cycle95-20260626-210628}"
RESUME="${RESUME:-$PREVIOUS_OUT/latest.pt}"

SCORE_LABEL="${SCORE_KOMI//./p}"
TEMP_LABEL="${TEMPERATURE//./p}"
MID_TEMP_LABEL="${MID_TEMPERATURE//./p}"
LATE_TEMP_LABEL="${LATE_TEMPERATURE//./p}"
LR_LABEL="${LEARNING_RATE//./p}"
TRAIN_OUT="${TRAIN_OUT:-artifacts/multiworker-13x13-cont-dualgpu-4x64-history2-autokomi-start${SCORE_LABEL}-ladder10p5-minpass${MIN_PASS_MOVE}-lr${LR_LABEL}-cleanup-finalboard-margin0p2-${SIMULATIONS}sims-max${MAX_MOVES}-temp${TEMP_LABEL}-moves${TEMPERATURE_MOVES}-mid${MID_TEMP_LABEL}-until${MID_TEMPERATURE_UNTIL}-late${LATE_TEMP_LABEL}-${WORKERS}w-after-temp0p7-${RUN_ID}}"
LOG="$TRAIN_OUT/train.log"
MONITOR="$TRAIN_OUT/gpu_monitor.csv"
STATUS_LOG="$TRAIN_OUT/status_3h.log"

mkdir -p "$TRAIN_OUT"
cat > "$TRAIN_OUT/run_notes.md" <<NOTES
# 13x13 final-board continuation: opening temperature 0.6

- created_time: $(date --iso-8601=seconds)
- source_commit: ${SOURCE_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || cat SOURCE_COMMIT 2>/dev/null || echo unknown)}
- purpose: continue the active 13x13 final-board-head line while lowering opening move temperature from 0.7 to 0.6
- previous_output: $PREVIOUS_OUT
- resume_checkpoint: $RESUME
- board_size: $BOARD_SIZE
- model: 4 residual blocks x 64 channels
- history_moves: 2
- input_komi: false
- komi_metadata: 0.5
- score_komi: $SCORE_KOMI
- score_komi_ladder: $SCORE_KOMI_LADDER
- score_komi_adjust_window: $SCORE_KOMI_ADJUST_WINDOW
- score_komi_adjust_threshold: $SCORE_KOMI_ADJUST_THRESHOLD
- terminal_dead_stone_cleanup: true
- score_margin_reward_scale: 0.2
- final_board_loss_weight: $FINAL_BOARD_LOSS_WEIGHT
- simulations: $SIMULATIONS
- workers: $WORKERS
- games_per_worker: $GAMES_PER_WORKER
- games_per_cycle: $((WORKERS * GAMES_PER_WORKER))
- max_moves: $MAX_MOVES
- min_pass_move: $MIN_PASS_MOVE
- train_steps_per_cycle: $TRAIN_STEPS_PER_CYCLE
- batch_size: $BATCH_SIZE
- replay_size: $REPLAY_SIZE
- learning_rate: $LEARNING_RATE
- c_puct: 1.5
- root_dirichlet_alpha: $ROOT_DIRICHLET_ALPHA
- root_noise_fraction: $ROOT_NOISE_FRACTION
- root_policy_temperature: $ROOT_POLICY_TEMPERATURE
- move_temperature_opening: $TEMPERATURE for moves [0, $TEMPERATURE_MOVES)
- move_temperature_middle: $MID_TEMPERATURE for moves [$TEMPERATURE_MOVES, $MID_TEMPERATURE_UNTIL)
- move_temperature_late: $LATE_TEMPERATURE for moves [$MID_TEMPERATURE_UNTIL, end)
- checkpoint_every: 10
- record_every: 5
- full_trace_every: 20
- full_trace_games: 5
- trace_top_actions_limit: 5
- full_trace_light_top5_search_tree: true, stored as top5_search_tree when root_search_visits >= 2
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

args=(
  -u -m diamondgo.multiworker_train_dualgpu
  --json
  --board-size "$BOARD_SIZE"
  --out-dir "$TRAIN_OUT"
  --device cuda:0
  --selfplay-devices cuda:0,cuda:1
  --rules sgfmill
  --komi 0.5
  --score-komi "$SCORE_KOMI"
  --no-input-komi
  --history-moves 2
  --terminal-dead-stone-cleanup
  --score-margin-reward-scale 0.2
  --final-board-loss-weight "$FINAL_BOARD_LOSS_WEIGHT"
  --score-komi-ladder "$SCORE_KOMI_LADDER"
  --score-komi-adjust-window "$SCORE_KOMI_ADJUST_WINDOW"
  --score-komi-adjust-threshold "$SCORE_KOMI_ADJUST_THRESHOLD"
  --time-limit-minutes "$TIME_LIMIT_MINUTES"
  --cycles 1000000
  --workers "$WORKERS"
  --games-per-worker "$GAMES_PER_WORKER"
  --max-moves "$MAX_MOVES"
  --min-pass-move "$MIN_PASS_MOVE"
  --simulations "$SIMULATIONS"
  --train-steps-per-cycle "$TRAIN_STEPS_PER_CYCLE"
  --batch-size "$BATCH_SIZE"
  --replay-size "$REPLAY_SIZE"
  --channels 64
  --residual-blocks 4
  --learning-rate "$LEARNING_RATE"
  --weight-decay 0.0001
  --c-puct 1.5
  --temperature "$TEMPERATURE"
  --temperature-moves "$TEMPERATURE_MOVES"
  --mid-temperature "$MID_TEMPERATURE"
  --mid-temperature-until "$MID_TEMPERATURE_UNTIL"
  --late-temperature "$LATE_TEMPERATURE"
  --root-dirichlet-alpha "$ROOT_DIRICHLET_ALPHA"
  --root-noise-fraction "$ROOT_NOISE_FRACTION"
  --root-policy-temperature "$ROOT_POLICY_TEMPERATURE"
  --augment-dihedral
  --checkpoint-every 10
  --early-checkpoint-cycles 50
  --early-checkpoint-every 5
  --record-every 5
  --full-trace-every 20
  --full-trace-games 5
  --trace-top-actions-limit 5
  --resume "$RESUME"
)

PYTHONPATH=src /root/miniconda3/bin/python "${args[@]}" > "$LOG" 2>&1
