set -euo pipefail

cd /root/diamondgo

TRAIN_PID="${1:?usage: finalize_fresh_nokomi_4x64_score2p5_noise_aug_2h.sh TRAIN_PID}"
TRAIN_OUT="artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605"
EVAL_OUT="artifacts/eval-suite-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605"
TACTICAL_OUT="artifacts/tactical-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605"
LOG="$TRAIN_OUT/finalize.log"

mkdir -p "$TRAIN_OUT" "$EVAL_OUT" "$TACTICAL_OUT"
{
  echo "finalizer_started_time=$(date --iso-8601=seconds)"
  echo "waiting_for_train_pid=$TRAIN_PID"
} >> "$LOG"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done

{
  echo "training_pid_finished_time=$(date --iso-8601=seconds)"
  echo "status=starting eval suite"
} >> "$LOG"

PYTHONPATH=src /root/miniconda3/bin/python -u -m diamondgo.eval_suite \
  --json \
  --checkpoint-dir "$TRAIN_OUT/checkpoints" \
  --out-dir "$EVAL_OUT" \
  --steps 50,200,500 \
  --opponents initial,previous \
  --games 20 \
  --simulations 100 \
  --max-moves 120 \
  --sample-games 2 \
  --include-latest "$TRAIN_OUT/latest.pt" \
  --device cuda \
  > "$EVAL_OUT/eval_suite.log" 2>&1

{
  echo "status=starting tactical eval"
} >> "$LOG"

PYTHONPATH=src /root/miniconda3/bin/python -u -m diamondgo.tactical_eval \
  --json \
  --checkpoint "$TRAIN_OUT/latest.pt" \
  --out-dir "$TACTICAL_OUT" \
  --simulations 100 \
  --device cuda \
  > "$TACTICAL_OUT/tactical_eval.log" 2>&1

{
  echo "finalizer_finished_time=$(date --iso-8601=seconds)"
  echo "eval_out=$EVAL_OUT"
  echo "tactical_out=$TACTICAL_OUT"
} >> "$LOG"
