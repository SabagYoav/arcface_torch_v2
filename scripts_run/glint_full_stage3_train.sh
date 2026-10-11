#!/bin/bash
# Stage 3 of glint_full_finetune_pipeline.sh (runs INSIDE the devcontainer):
# fine-tune R50 with ArcFace on mixed full/ROI-0.15 over the full glint360k.
# GPUs 0-2 (GPU 3 is reserved for the swin_base clip_ratio_15 20->30 epoch
# extension, scripts_run/swin_ext30_launch.sh).
set -euo pipefail
cd /home/yoav/arcface_torch_v2
CONFIG=${CONFIG:-configs/finetune_r50_arcface_mixed_roi15_glint_full.py}

CUDA_VISIBLE_DEVICES=0,1,2 /home/yoav/.local/bin/torchrun --nproc_per_node=3 --master_port=12611 \
    train_v5_pairs.py --config "$CONFIG"
