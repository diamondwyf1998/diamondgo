set -euo pipefail

cd /root/diamondgo

TRAIN_OUT="artifacts/multiworker-9x9-cont-nokomi-4x64-score5p5-margin0p2-300sims-max150-6h-20260606"
QUEUE_LOG="$TRAIN_OUT/queue.log"
RUN_SCRIPT="/root/diamondgo/tools/server/run_cont_nokomi_4x64_score5p5_margin0p2_300sims_6h.sh"
FINALIZE_SCRIPT="/root/diamondgo/tools/server/finalize_cont_nokomi_4x64_score5p5_margin0p2_300sims_6h.sh"

mkdir -p "$TRAIN_OUT"
{
  echo "queue_started_time=$(date --iso-8601=seconds)"
  echo "reason=wait for existing score6p5 finalizer/eval/pairwise work before starting new training"
} >> "$QUEUE_LOG"

while true; do
  if pgrep -af "finalize_fresh_nokomi_4x64_score6p5_200sims_noise_aug_5h|eval-suite-fresh-nokomi-4x64-score6p5|pairwise_50cycle_extra6" >/tmp/diamondgo_waiting_processes.txt; then
    {
      echo "waiting_time=$(date --iso-8601=seconds)"
      cat /tmp/diamondgo_waiting_processes.txt
    } >> "$QUEUE_LOG"
    sleep 120
    continue
  fi
  break
done

{
  echo "training_start_time=$(date --iso-8601=seconds)"
  echo "run_script=$RUN_SCRIPT"
  echo "finalize_script=$FINALIZE_SCRIPT"
} >> "$QUEUE_LOG"

bash "$RUN_SCRIPT" &
TRAIN_PID=$!
echo "train_shell_pid=$TRAIN_PID" >> "$QUEUE_LOG"

nohup bash "$FINALIZE_SCRIPT" "$TRAIN_PID" \
  > "$TRAIN_OUT/finalize.nohup.log" 2>&1 &
echo "finalizer_pid=$!" >> "$QUEUE_LOG"

wait "$TRAIN_PID"
echo "training_shell_finished_time=$(date --iso-8601=seconds)" >> "$QUEUE_LOG"
