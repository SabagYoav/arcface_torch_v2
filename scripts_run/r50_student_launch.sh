#!/bin/bash
# iResNet-50 student, CLIP distillation, full glint360k, ROI 0.15, 10 epochs, GPU 0.
# Runs on the HOST in tmux; trains in the devcontainer. Resumable: re-running
# continues from the output folder's checkpoint_gpu_0.pt.
#   tmux new-session -d -s r50_student 'bash scripts_run/r50_student_launch.sh'
set -uo pipefail
OUT=/home/yoav/DEV/arcface_torch_v2/work_dirs/exp_clip_r50_vs_r50_glint/clip_ratio_15
LOGFILE=$OUT/launch.log
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOGFILE"; }
c=$(docker ps --format '{{.Names}} {{.Image}}' | awk '/vsc-arcface_torch_v2/ {print $1; exit}')
if [ -z "$c" ]; then log "ERROR: arcface devcontainer not running"; exit 1; fi
log "Starting r50 student (container $c, GPU 0)"
docker exec -u 1016 "$c" bash -c "cd /home/yoav/arcface_torch_v2 && CUDA_VISIBLE_DEVICES=0 \
    python3 train_v4_clip.py --config configs/variants_clip_r50_student_ratio_15.py" >> "$LOGFILE" 2>&1
log "train_v4_clip.py exited with status $?"
