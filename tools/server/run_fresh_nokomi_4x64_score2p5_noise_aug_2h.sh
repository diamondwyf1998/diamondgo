set -euo pipefail

cd /root/diamondgo

TRAIN_OUT="artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605"
LOG="$TRAIN_OUT/train.log"
MONITOR="$TRAIN_OUT/gpu_monitor.csv"

mkdir -p "$TRAIN_OUT"
cat > "$TRAIN_OUT/run_notes.md" <<NOTES
# Fresh No-Komi-Input 4x64 Run

- created_time: $(date --iso-8601=seconds)
- fresh_start: true
- reason: previous continuation runs remained much weaker/skewed than the user's earlier CPU runs
- model_input_komi: false
- score_komi: 2.5
- channels: 64
- residual_blocks: 4
- root_noise: Dirichlet alpha 0.15, fraction 0.25
- root_policy_temperature: 1.1
- move_temperature: 1.0 for first 16 moves, then 0.25
- augmentation: random dihedral board symmetries during training
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
  --device cuda \
  --rules sgfmill \
  --komi 0.5 \
  --score-komi 2.5 \
  --no-input-komi \
  --time-limit-minutes 120 \
  --cycles 100000 \
  --workers 8 \
  --games-per-worker 4 \
  --max-moves 120 \
  --simulations 100 \
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
