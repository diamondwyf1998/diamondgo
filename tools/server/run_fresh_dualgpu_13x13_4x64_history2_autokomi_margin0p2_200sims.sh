#!/usr/bin/env bash
set -euo pipefail

cd /root/diamondgo

# 30+ spawned self-play workers can exceed the default 1024 open-file limit when
# PyTorch shares model tensors with subprocesses.
ulimit -n "${ULIMIT_NOFILE:-65535}" || true

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
BOARD_SIZE="${BOARD_SIZE:-13}"
SCORE_KOMI="${SCORE_KOMI:-6.5}"
SCORE_KOMI_LADDER="${SCORE_KOMI_LADDER:-2.5,4.5,6.5,7.5,8.5}"
SCORE_KOMI_ADJUST_WINDOW="${SCORE_KOMI_ADJUST_WINDOW:-3}"
SCORE_KOMI_ADJUST_THRESHOLD="${SCORE_KOMI_ADJUST_THRESHOLD:-0.75}"
SIMULATIONS="${SIMULATIONS:-200}"
MAX_MOVES="${MAX_MOVES:-250}"
WORKERS="${WORKERS:-30}"
GAMES_PER_WORKER="${GAMES_PER_WORKER:-8}"
TIME_LIMIT_MINUTES="${TIME_LIMIT_MINUTES:-0}"
TRAIN_STEPS_PER_CYCLE="${TRAIN_STEPS_PER_CYCLE:-64}"
BATCH_SIZE="${BATCH_SIZE:-256}"
REPLAY_SIZE="${REPLAY_SIZE:-100000}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TEMPERATURE_MOVES="${TEMPERATURE_MOVES:-16}"
LATE_TEMPERATURE="${LATE_TEMPERATURE:-0.2}"
ROOT_DIRICHLET_ALPHA="${ROOT_DIRICHLET_ALPHA:-0.15}"
ROOT_NOISE_FRACTION="${ROOT_NOISE_FRACTION:-0.25}"
ROOT_POLICY_TEMPERATURE="${ROOT_POLICY_TEMPERATURE:-1.1}"
RESUME="${RESUME:-}"

SCORE_LABEL="${SCORE_KOMI//./p}"
TEMP_LABEL="${TEMPERATURE//./p}"
LATE_TEMP_LABEL="${LATE_TEMPERATURE//./p}"
TRAIN_OUT="${TRAIN_OUT:-artifacts/multiworker-13x13-fresh-dualgpu-4x64-history2-autokomi-start${SCORE_LABEL}-cleanup-margin0p2-${SIMULATIONS}sims-max${MAX_MOVES}-temp${TEMP_LABEL}-late${LATE_TEMP_LABEL}-${WORKERS}w-${RUN_ID}}"
LOG="$TRAIN_OUT/train.log"
MONITOR="$TRAIN_OUT/gpu_monitor.csv"
STATUS_LOG="$TRAIN_OUT/status_3h.log"

mkdir -p "$TRAIN_OUT"
cat > "$TRAIN_OUT/run_notes.md" <<NOTES
# Fresh Dual-GPU 13x13 4x64, History-2, Auto-Komi Start $SCORE_KOMI, Cleanup, Margin Reward 0.2

- created_time: $(date --iso-8601=seconds)
- source_commit: $(git rev-parse --short HEAD 2>/dev/null || cat SOURCE_COMMIT 2>/dev/null || echo unknown)
- purpose: first 13x13 experiment while preserving the current 4x64 history-2 training architecture and core tricks
- fresh_start: $([ -z "$RESUME" ] && echo true || echo false)
- resume_checkpoint: ${RESUME:-none}
- comparison_warning: 9x9 checkpoints cannot be resumed directly because 13x13 changes action count from 82 to 170
- board_size: $BOARD_SIZE
- action_count: 170 for 13x13, including pass
- model_input_komi: false
- history_moves: 2
- input_planes: 5
- komi_metadata: 0.5
- initial_score_komi: $SCORE_KOMI
- dynamic_score_komi_ladder: $SCORE_KOMI_LADDER
- dynamic_score_komi_rule: if rolling black win rate is above threshold, increase score komi one ladder step; if rolling white win rate is above threshold, decrease one ladder step
- dynamic_score_komi_adjust_window: $SCORE_KOMI_ADJUST_WINDOW cycles
- dynamic_score_komi_adjust_threshold: $SCORE_KOMI_ADJUST_THRESHOLD
- terminal_dead_stone_cleanup: true
- cleanup_scope: conservative obvious-dead cleanup only; not full life-and-death solving
- score_margin_reward_scale: 0.2
- value_target_formula_when_margin_reward_enabled: sign(score_margin) * (2/5 + min(abs(score_margin) ** 0.25 / 5 * scale, 3/5))
- channels: 64
- residual_blocks: 4
- parameters_expected_13x13: about 367717
- trainer_device: cuda:0
- selfplay_devices: cuda:0,cuda:1
- workers: $WORKERS
- games_per_worker: $GAMES_PER_WORKER
- games_per_cycle: $((WORKERS * GAMES_PER_WORKER))
- max_moves: $MAX_MOVES
- selfplay_simulations: $SIMULATIONS
- train_steps_per_cycle: $TRAIN_STEPS_PER_CYCLE
- batch_size: $BATCH_SIZE
- replay_size: $REPLAY_SIZE
- optimizer: AdamW, learning rate 0.001, weight decay 0.0001
- c_puct: 1.5
- root_noise: Dirichlet alpha $ROOT_DIRICHLET_ALPHA, fraction $ROOT_NOISE_FRACTION
- root_policy_temperature: $ROOT_POLICY_TEMPERATURE
- move_temperature: $TEMPERATURE for first $TEMPERATURE_MOVES moves, then $LATE_TEMPERATURE
- augmentation: random dihedral board symmetries during training
- checkpoint_every: 10 cycles after the early dense window
- early_checkpoint_cycles: 50
- early_checkpoint_every: 5
- complete_sgf_trace_archive: every 5 cycles in cycle-records/
- record_every: 5
- full_search_tree_trace: every 20 cycles only, first 5 games
- full_trace_every: 20
- full_trace_games: 5
- non_full_trace_top_actions_limit: 5
- time_limit_minutes: $TIME_LIMIT_MINUTES
- nofile_limit: $(ulimit -n)
- operational_note: 30-worker 13x13 stress failed at default nofile=1024 and passed after raising nofile to 65535
- old scripts/checkpoints: preserved; this script writes a new 13x13 output directory
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
  --score-komi-ladder "$SCORE_KOMI_LADDER"
  --score-komi-adjust-window "$SCORE_KOMI_ADJUST_WINDOW"
  --score-komi-adjust-threshold "$SCORE_KOMI_ADJUST_THRESHOLD"
  --time-limit-minutes "$TIME_LIMIT_MINUTES"
  --cycles 1000000
  --workers "$WORKERS"
  --games-per-worker "$GAMES_PER_WORKER"
  --max-moves "$MAX_MOVES"
  --simulations "$SIMULATIONS"
  --train-steps-per-cycle "$TRAIN_STEPS_PER_CYCLE"
  --batch-size "$BATCH_SIZE"
  --replay-size "$REPLAY_SIZE"
  --channels 64
  --residual-blocks 4
  --learning-rate 0.001
  --weight-decay 0.0001
  --c-puct 1.5
  --temperature "$TEMPERATURE"
  --temperature-moves "$TEMPERATURE_MOVES"
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
)

if [ -n "$RESUME" ]; then
  args+=(--resume "$RESUME")
fi

PYTHONPATH=src /root/miniconda3/bin/python "${args[@]}" > "$LOG" 2>&1
