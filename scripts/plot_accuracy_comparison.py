import json
import re
import matplotlib.pyplot as plt

# Load baseline fullface encoder accuracy
with open("work_dirs/baseline_fullface_encoder_accuracy.json") as f:
    baseline_data = json.load(f)

baseline_ratios = sorted([int(k) for k in baseline_data.keys()], reverse=True)
baseline_accs = [baseline_data[str(r)]["best_acc"] for r in baseline_ratios]

# Parse CLIP training log
clip_data = {}
# with open("training_multi_loops_log_clip.txt") as f:
#     for line in f:
#         m = re.search(r"Completed training for ratio_(\d+) with results: \{'best_acc': ([0-9.]+)\}", line)
#         if m:
#             clip_data[int(m.group(1))] = float(m.group(2))
with open("work_dirs/clip_encoder_accuracy.json") as f:
    clip_data = json.load(f)

clip_ratios = sorted([int(k) for k in clip_data.keys()], reverse=True)
clip_accs = [clip_data[str(r)]["best_acc"] if isinstance(clip_data[str(r)], dict) else clip_data[str(r)] for r in clip_ratios]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(baseline_ratios, baseline_accs, 'o-', label='Baseline (ArcFace)', linewidth=2, markersize=8)
ax.plot(clip_ratios, clip_accs, 's-', label='CLIP-trained (InfoNCE)', linewidth=2, markersize=8)

ax.set_xlabel('ROI Ratio (%)', fontsize=13)
ax.set_ylabel('Verification Accuracy', fontsize=13)
ax.set_title('Verification Accuracy vs ROI Ratio', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(set(baseline_ratios + clip_ratios), reverse=True))
ax.invert_xaxis()

# Annotate points
for r, a in zip(baseline_ratios, baseline_accs):
    ax.annotate(f'{a:.4f}', (r, a), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
for r, a in zip(clip_ratios, clip_accs):
    ax.annotate(f'{a:.4f}', (r, a), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('work_dirs/accuracy_comparison.png', dpi=150)
plt.show()
print("Saved to work_dirs/accuracy_comparison.png")
