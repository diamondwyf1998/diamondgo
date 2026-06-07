set -euo pipefail

cd /root/diamondgo

OLD_TRAIN_PID="${1:?usage: queue_cont_dualgpu_2x96_score4p5_margin0p2_250sims_3h_after18h.sh OLD_TRAIN_PID [OLD_FINALIZER_PID]}"
OLD_FINALIZER_PID="${2:-}"
TRAIN_OUT="artifacts/multiworker-9x9-cont-dualgpu-2x96-score4p5-margin0p2-250sims-max150-3h-after18h-20260607"
BASE_OUT="artifacts/multiworker-9x9-fresh-dualgpu-2x96-score4p5-margin0p2-250sims-max150-18h-20260607"
QUEUE_LOG="$TRAIN_OUT/queue.log"
RUN_SCRIPT="/root/diamondgo/tools/server/run_cont_dualgpu_2x96_score4p5_margin0p2_250sims_3h_after18h.sh"
FINALIZE_SCRIPT="/root/diamondgo/tools/server/finalize_cont_dualgpu_2x96_score4p5_margin0p2_250sims_3h_after18h.sh"

mkdir -p "$TRAIN_OUT"
{
  echo "queue_started_time=$(date --iso-8601=seconds)"
  echo "reason=wait for 18h dual-GPU training, then immediately start a 3h continuation before final eval"
  echo "old_train_pid=$OLD_TRAIN_PID"
  echo "old_finalizer_pid=${OLD_FINALIZER_PID:-none}"
  echo "base_out=$BASE_OUT"
  echo "continuation_out=$TRAIN_OUT"
} >> "$QUEUE_LOG"

if [[ -n "$OLD_FINALIZER_PID" ]] && ps -p "$OLD_FINALIZER_PID" -o cmd= 2>/dev/null | grep -q "finalize_fresh_dualgpu_2x96_score4p5_margin0p2_250sims_18h.sh"; then
  {
    echo "old_finalizer_stop_time=$(date --iso-8601=seconds)"
    echo "old_finalizer_stop_reason=replaced by continuation queue so eval does not compete with the +3h training"
  } >> "$QUEUE_LOG"
  kill "$OLD_FINALIZER_PID" 2>/dev/null || true
fi

while kill -0 "$OLD_TRAIN_PID" 2>/dev/null; do
  sleep 60
done

{
  echo "old_training_finished_time=$(date --iso-8601=seconds)"
  echo "resume_checkpoint=$BASE_OUT/latest.pt"
  echo "training_start_time=$(date --iso-8601=seconds)"
  echo "run_script=$RUN_SCRIPT"
  echo "finalize_script=$FINALIZE_SCRIPT"
} >> "$QUEUE_LOG"

RESUME_CHECKPOINT="$BASE_OUT/latest.pt" bash "$RUN_SCRIPT" &
TRAIN_PID=$!
echo "$TRAIN_PID" > artifacts/dualgpu_2x96_score4p5_250sims_3h_after18h.train.pid
echo "train_shell_pid=$TRAIN_PID" >> "$QUEUE_LOG"

nohup bash "$FINALIZE_SCRIPT" "$TRAIN_PID" \
  > "$TRAIN_OUT/finalize.nohup.log" 2>&1 &
FINALIZER_PID=$!
echo "$FINALIZER_PID" > artifacts/dualgpu_2x96_score4p5_250sims_3h_after18h.finalizer.pid
echo "finalizer_pid=$FINALIZER_PID" >> "$QUEUE_LOG"

wait "$TRAIN_PID"
echo "continuation_training_shell_finished_time=$(date --iso-8601=seconds)" >> "$QUEUE_LOG"
