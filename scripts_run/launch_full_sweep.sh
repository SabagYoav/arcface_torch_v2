#!/bin/bash
# Waits for the eye-center metadata job (data_scratches/build_eye_center_metadata.py,
# 4 GPU shards) to finish, merges its output, then launches the full glint360k
# ROI sweep (training_multi_loops.py) with on-the-fly cropping + LFW validation.
# Meant to run inside a detached tmux session so it survives client disconnects.
set -uo pipefail

cd /home/yoav/arcface_torch_v2
LOGFILE=/home/yoav/arcface_torch_v2/full_sweep_launch.log
METADATA_LOG=/home/yoav/arcface_torch_v2/eye_center_metadata_build.log
CUDNN_LIB=/home/yoav/.local/lib/python3.10/site-packages/nvidia/cudnn/lib

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOGFILE"; }

log "Waiting for eye-center metadata job to finish..."
while pgrep -f "data_scratches/build_eye_center_metadata.py" > /dev/null; do
  sleep 60
done
log "Metadata job process(es) exited."

for shard in 0 1 2 3; do
  if ! grep -q "\[shard $shard\] DONE" "$METADATA_LOG"; then
    log "ERROR: shard $shard did not report DONE in $METADATA_LOG, aborting."
    exit 1
  fi
done
log "All 4 shards confirmed DONE."

log "Merging shard outputs into final metadata JSON..."
LD_LIBRARY_PATH="$CUDNN_LIB" python3 data_scratches/build_eye_center_metadata.py --merge 2>&1 | tee -a "$LOGFILE"

log "Starting full ROI sweep (CUDA_VISIBLE_DEVICES=0, single GPU)."
export CUDA_VISIBLE_DEVICES=0
LD_LIBRARY_PATH="$CUDNN_LIB" python3 training_multi_loops.py 2>&1 | tee -a "$LOGFILE"
status=$?
log "training_multi_loops.py exited with status $status"
