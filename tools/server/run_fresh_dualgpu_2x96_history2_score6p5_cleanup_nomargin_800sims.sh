set -euo pipefail

cd /root/diamondgo

RUN_ID="${RUN_ID:-20260616}"
TIME_LIMIT_MINUTES="${TIME_LIMIT_MINUTES:-1440}"
SCORE_KOMI="${SCORE_KOMI:-6.5}"
SIMULATIONS="${SIMULATIONS:-800}"
SCORE_LABEL="${SCORE_KOMI//./p}"
TRAIN_OUT="${TRAIN_OUT:-artifacts/multiworker-9x9-fresh-dualgpu-2x96-history2-score${SCORE_LABEL}-cleanup-nomargin-${SIMULATIONS}sims-max150-${TIME_LIMIT_MINUTES}m-${RUN_ID}}"
LOG="$TRAIN_OUT/train.log"
MONITOR="$TRAIN_OUT/gpu_monitor.csv"
STATUS_LOG="$TRAIN_OUT/status_3h.log"

mkdir -p "$TRAIN_OUT"
cat > "$TRAIN_OUT/run_notes.md" <<NOTES
# Fresh Dual-GPU 2x96, History-2 Input, Score Komi $SCORE_KOMI, Cleanup, No Margin Reward, $SIMULATIONS Sims

- created_time: $(date --iso-8601=seconds)
- source_commit: $(git rev-parse --short HEAD 2>/dev/null || cat SOURCE_COMMIT 2>/dev/null || echo unknown)
- purpose: comparison experiment; add only previous-two-move input planes on top of the selected cleanup baseline
- fresh_start: true
- resume_checkpoint: none
- comparison_warning: this run changes the neural input shape; compare by positions/games/wall-clock/eval, not only raw cycle ID
- model_input_komi: false
- history_moves: 2
- input_planes: 5
- input_plane_order: own stones, opponent stones, to-play black plane, previous move, previous-previous move
- pass_history_encoding: pass moves are encoded as all-zero history planes
- komi_metadata: 0.5
- score_komi: $SCORE_KOMI
- terminal_dead_stone_cleanup: true
- cleanup_scope: remove only conservatively detected obvious dead groups; edge deaths, seki, ko, and complex life-and-death are not solved
- score_margin_reward_scale: 0.0
- value_target_formula: pure win/loss target, no score-margin bonus
- channels: 96
- residual_blocks: 2
- trainer_device: cuda:0
- selfplay_devices: cuda:0,cuda:1
- workers: 12
- games_per_worker: 8
- games_per_cycle: 96
- max_moves: 150
- selfplay_simulations: $SIMULATIONS
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
- full_search_tree_trace: every 20 cycles only, first 5 games
- full_trace_every: 20
- full_trace_games: 5
- non_full_trace_top_actions_limit: 5
- time_limit_minutes: $TIME_LIMIT_MINUTES
- old scripts/checkpoints: preserved; this run uses a new output directory and the optional --history-moves feature
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
  --device cuda:0 \
  --selfplay-devices cuda:0,cuda:1 \
  --rules sgfmill \
  --komi 0.5 \
  --score-komi "$SCORE_KOMI" \
  --no-input-komi \
  --history-moves 2 \
  --terminal-dead-stone-cleanup \
  --score-margin-reward-scale 0.0 \
  --time-limit-minutes "$TIME_LIMIT_MINUTES" \
  --cycles 100000 \
  --workers 12 \
  --games-per-worker 8 \
  --max-moves 150 \
  --simulations "$SIMULATIONS" \
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
  --full-trace-every 20 \
  --full-trace-games 5 \
  --trace-top-actions-limit 5 \
  > "$LOG" 2>&1
