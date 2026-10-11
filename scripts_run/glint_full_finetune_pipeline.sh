#!/bin/bash
# Unattended pipeline for fine-tuning R50 with ArcFace on the FULL glint360k
# (mixed full-face + ROI-0.15). Runs on the HOST inside tmux, so it survives
# client disconnects; GPU stages run in the devcontainer via `docker exec`.
#   1. extract all glint360k identities from the archive  (host, ~2h)
#   2. rebuild full eye-center metadata (buffalo_l, 4 GPUs) (container, ~6h)
#   3. train (torchrun, 4 GPUs)                             (container)
# Each stage writes a done-marker, so re-running this script resumes.
#
# Launch:  tmux new-session -d -s glint_full 'bash scripts_run/glint_full_finetune_pipeline.sh'
set -uo pipefail

REPO=/home/yoav/DEV/arcface_torch_v2
DATA=/home/yoav/DATA
ARCHIVE=/mnt/ib_video/DATA/face_ReId/glint360k_download/glint360k
IMAGES=$DATA/glint360k_download/glint360k_images
STATE=$DATA/glint360k_download/pipeline_state
LOGFILE=$REPO/work_dirs/exp_finetune_r50_arcface/glint_full_pipeline.log
mkdir -p "$STATE" "$(dirname "$LOGFILE")"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOGFILE"; }

container() {
    # Resolve at each stage: a devcontainer rebuild changes the name.
    docker ps --format '{{.Names}} {{.Image}}' | awk '/vsc-arcface_torch_v2/ {print $1; exit}'
}

in_container() {
    local c; c=$(container)
    if [ -z "$c" ]; then log "ERROR: arcface devcontainer not running"; exit 1; fi
    docker exec -u 1016 "$c" bash "$@"
}

log "=== pipeline start ==="

if [ ! -f "$STATE/1_extract.done" ]; then
    log "Stage 1: extracting all glint360k identities -> $IMAGES"
    (cd "$ARCHIVE" && cat glint360k_0[0-6] | pigz -dc | python3 -I "$REPO/data_scratches/extract_glint360k_all.py" \
        "$DATA/glint360k_arch_comparison/sample_manifest.json" "$IMAGES" "$STATE/extract_counts.json") \
        >> "$LOGFILE" 2>&1
    if [ $? -ne 0 ] || ! grep -q "^DONE" "$LOGFILE"; then log "ERROR: stage 1 failed"; exit 1; fi
    log "Stage 1 done: $(ls "$IMAGES" | wc -l) identity dirs"
    touch "$STATE/1_extract.done"
fi

if [ ! -f "$STATE/2_metadata.done" ]; then
    log "Stage 2: eye-center metadata"
    in_container /home/yoav/arcface_torch_v2/scripts_run/glint_full_stage2_metadata.sh >> "$LOGFILE" 2>&1
    if [ $? -ne 0 ]; then log "ERROR: stage 2 failed"; exit 1; fi
    touch "$STATE/2_metadata.done"
    log "Stage 2 done"
fi

if [ ! -f "$STATE/3_train.done" ]; then
    log "Stage 3: training"
    in_container /home/yoav/arcface_torch_v2/scripts_run/glint_full_stage3_train.sh >> "$LOGFILE" 2>&1
    if [ $? -ne 0 ]; then log "ERROR: stage 3 failed"; exit 1; fi
    touch "$STATE/3_train.done"
    log "Stage 3 done"
fi

log "=== pipeline finished ==="
