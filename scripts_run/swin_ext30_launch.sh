#!/bin/bash
# Extend swin_base CLIP ratio-0.15 (full glint360k) from 20 to 30 epochs on GPU 3.
# Runs on the HOST inside tmux. Waits until the full-glint eye-center metadata
# exists (stage 2 of glint_full_finetune_pipeline.sh), then trains in the
# devcontainer. Resumable: re-running resumes from the new folder's checkpoint.
#
# Launch:  tmux new-session -d -s swin_ext30 'bash scripts_run/swin_ext30_launch.sh'
set -uo pipefail

REPO=/home/yoav/DEV/arcface_torch_v2
STATE=/home/yoav/DATA/glint360k_download/pipeline_state
OUT=$REPO/work_dirs/exp_clip_r50_vs_swin_glint/clip_ratio_15_ext30
LOGFILE=$OUT/launch.log
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOGFILE"; }

log "Waiting for full-glint metadata ($STATE/2_metadata.done)..."
while [ ! -f "$STATE/2_metadata.done" ]; do sleep 300; done

c=$(docker ps --format '{{.Names}} {{.Image}}' | awk '/vsc-arcface_torch_v2/ {print $1; exit}')
if [ -z "$c" ]; then log "ERROR: arcface devcontainer not running"; exit 1; fi

log "Metadata ready. Starting swin_base 20->30 epoch extension on GPU 3 (container $c)."
docker exec -u 1016 "$c" bash -c "cd /home/yoav/arcface_torch_v2 && CUDA_VISIBLE_DEVICES=3 \
    python3 train_v4_clip.py --config configs/variants_clip_swin_base_ratio_15_ext30.py" >> "$LOGFILE" 2>&1
log "train_v4_clip.py exited with status $?"
