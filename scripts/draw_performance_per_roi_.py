import re
import pandas as pd
from pathlib import Path

log_path = Path("training_multi_loops_log_clip.txt")  # change to your file

pattern = re.compile(
    r"Completed training for ratio_(\d+).*best_acc': ([0-9.]+)"
)

rows = []

with open(log_path, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            roi = int(m.group(1))
            acc = float(m.group(2))
            rows.append({"roi": roi, "accuracy": acc})

df = pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)

print(df)

import matplotlib.pyplot as plt

plt.figure()

plt.plot(df.roi, df.accuracy, marker="o", linewidth=2)

plt.gca().invert_xaxis()
plt.ylim(0,1)

plt.xlabel("ROI ratio (%)")
plt.ylabel("Verification Accuracy")
plt.title("ArcFace performance vs ROI size")
plt.grid()

plt.savefig("roi_sweep_performace.png")
plt.close()