#!/bin/bash
# One-shot: stop all training tmux sessions, verify GPU memory freed, then
# remove itself from crontab (self-cleaning, since cron has no year field for
# genuine one-time scheduling).
LOGFILE=/home/yoav/arcface_torch_v2/stop_all_training.log
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOGFILE"; }

log "=== stop_all_training.sh firing ==="
for s in $(tmux list-sessions -F '#{session_name}' 2>/dev/null); do
    log "Stopping tmux session: $s"
    tmux kill-session -t "$s" 2>&1 | tee -a "$LOGFILE"
done

sleep 2
log "GPU status after stop:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>&1 | tee -a "$LOGFILE"

# Remove this job from crontab so it only fires once.
crontab -l 2>/dev/null | grep -v "stop_all_training.sh" | crontab -
log "Removed self from crontab."
