set -euo pipefail

cd /root/diamondgo

TRAIN_PID="${1:?usage: finalize_cont_dualgpu_2x96_score6p5_cleanup_margin0p2_400sims_7d_after18hplus3h.sh TRAIN_PID}"
TRAIN_OUT="artifacts/multiworker-9x9-cont-dualgpu-2x96-score6p5-cleanup-margin0p2-400sims-max150-7d-after18hplus3h-20260607"
EVAL_OUT="artifacts/eval-suite-cont-dualgpu-2x96-score6p5-cleanup-margin0p2-400sims-train-100sims-eval-7d-after18hplus3h-20260607"
TACTICAL_OUT="artifacts/tactical-cont-dualgpu-2x96-score6p5-cleanup-margin0p2-400sims-train-100sims-eval-7d-after18hplus3h-20260607"
LOG="$TRAIN_OUT/finalize.log"

mkdir -p "$TRAIN_OUT" "$EVAL_OUT" "$TACTICAL_OUT"
{
  echo "finalizer_started_time=$(date --iso-8601=seconds)"
  echo "waiting_for_train_pid=$TRAIN_PID"
  echo "eval_note=standard 100-simulation eval after the cleanup-enabled 7d continuation; training self-play used 400 simulations"
  echo "eval_steps=50,200,500"
  echo "eval_opponents=initial,previous"
  echo "eval_games_per_match=20"
  echo "eval_sample_sgfs_per_match=2"
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
  --max-moves 150 \
  --sample-games 2 \
  --include-latest "$TRAIN_OUT/latest.pt" \
  --device cuda:0 \
  > "$EVAL_OUT/eval_suite.log" 2>&1

{
  echo "status=starting tactical eval"
} >> "$LOG"

PYTHONPATH=src /root/miniconda3/bin/python -u -m diamondgo.tactical_eval \
  --json \
  --checkpoint "$TRAIN_OUT/latest.pt" \
  --out-dir "$TACTICAL_OUT" \
  --simulations 100 \
  --device cuda:0 \
  > "$TACTICAL_OUT/tactical_eval.log" 2>&1

{
  echo "finalizer_finished_time=$(date --iso-8601=seconds)"
  echo "eval_out=$EVAL_OUT"
  echo "tactical_out=$TACTICAL_OUT"
} >> "$LOG"
