set -euo pipefail

cd /root/diamondgo

TRAIN_OUT="artifacts/multiworker-9x9-resume0p5-score2p5-100sims-120moves-1h-20260605"
RESUME="artifacts/multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605/latest.pt"

mkdir -p "$TRAIN_OUT"
cat > "$TRAIN_OUT/run_notes.md" <<NOTES
# Score Komi 2.5 Continuation

- created_time: $(date --iso-8601=seconds)
- resume: $RESUME
- reason: score_komi=6.5 produced persistent White self-play win rate above 90% by cycle 1064
- model_input_komi: 0.5
- score_komi: 2.5
- semantics: keep old model-input komi plane stable; change only terminal scoring/value labels
NOTES

PYTHONPATH=src /root/miniconda3/bin/python -u -m diamondgo.multiworker_train \
  --json \
  --resume "$RESUME" \
  --out-dir "$TRAIN_OUT" \
  --device cuda \
  --rules sgfmill \
  --komi 0.5 \
  --score-komi 2.5 \
  --time-limit-minutes 60 \
  --cycles 100000 \
  --workers 8 \
  --games-per-worker 4 \
  --max-moves 120 \
  --simulations 100 \
  --train-steps-per-cycle 64 \
  --batch-size 256 \
  --replay-size 50000 \
  --channels 32 \
  --residual-blocks 2 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --checkpoint-every 10 \
  > "$TRAIN_OUT/train.log" 2>&1
