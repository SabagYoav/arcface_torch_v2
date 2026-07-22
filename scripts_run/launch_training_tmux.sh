#!/bin/bash
# Waits for the /DATA/ROIs/ratio_100 scp copy to finish, then launches training.
# Meant to run inside a detached tmux session so it survives client disconnects.
set -uo pipefail

cd /home/yoav/arcface_torch_v2
LOGFILE=/home/yoav/arcface_torch_v2/training_tmux_launch.log
ROI_DIR=/DATA/ROIs/ratio_100

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOGFILE"; }

log "Waiting for scp into /home/yoav/DATA/ROIs to finish..."
# NOTE: use -x (exact process name) not -f (full cmdline) — a -f match on this
# literal string previously self-matched unrelated wait-loop shells whose own
# command text happened to contain it, causing an infinite false-positive wait.
while pgrep -x scp > /dev/null; do
  sleep 15
done
log "No scp process detected. Verifying copy is stable..."

# Confirm size is stable for a bit (guards against a new scp starting right after this check).
prev_size=$(du -sb "$ROI_DIR" | cut -f1)
sleep 30
curr_size=$(du -sb "$ROI_DIR" | cut -f1)
if [ "$prev_size" != "$curr_size" ]; then
  log "Size still changing ($prev_size -> $curr_size), waiting longer..."
  sleep 60
  while pgrep -x scp > /dev/null; do
    sleep 15
  done
fi

for split in train val test; do
  if [ ! -d "$ROI_DIR/$split" ]; then
    log "ERROR: missing split '$split' under $ROI_DIR, aborting."
    exit 1
  fi
  count=$(find "$ROI_DIR/$split" -mindepth 1 -maxdepth 1 -type d | wc -l)
  log "Split '$split': $count id dirs."
done

log "Starting training (CUDA_VISIBLE_DEVICES=0, single GPU)."
export CUDA_VISIBLE_DEVICES=0
python3 training_multi_loops.py 2>&1 | tee -a "$LOGFILE"
status=$?
log "training_multi_loops.py exited with status $status"
