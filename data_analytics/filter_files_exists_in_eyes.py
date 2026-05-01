import json
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# Inputs
# --------------------------------------------------
SRC_JSON = "data_analytics/test_set_pose_results.json"
OUT_JSON = "eyes_test_set_pose_results.json"

SRC_PREFIX = Path("/DATA_FP/glint360k/imageFolder_split_fullface/test")
DST_ROOT   = Path("/Downloads/Yoav/datasets/glint360k/imageFolder_split_narrow_eyes/test")

# --------------------------------------------------
# Load JSON
# --------------------------------------------------
with open(SRC_JSON, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries")

filtered = []
filtered_data = []
rel_path_list = []

# --------------------------------------------------
# Filter
# --------------------------------------------------
for item in tqdm(data, desc="Filtering"):
    src_path = Path(item["path"])

    # Ensure expected prefix exists
    try:
        rel_path = src_path.relative_to(SRC_PREFIX)
    except ValueError:
        # Path does not match expected source layout
        continue

    # Map to destination filesystem
    dst_path = DST_ROOT / rel_path

    if dst_path.exists():
        filtered.append(str(Path('test') / rel_path))
        new_item = item.copy()
        new_item["path"] = str(str(Path('test') / rel_path))
        filtered_data.append(new_item)
    rel_path_list.append(str(Path('test') / rel_path))
# --------------------------------------------------
# Save
# --------------------------------------------------
with open(OUT_JSON, "w") as f:
    json.dump(filtered_data, f, indent=2)
with open("rel_path_test_set_pose_results.txt", "w") as f:
    for p in rel_path_list:
        f.write(p + "\n")

print(f"Saved {len(filtered_data)} entries to {OUT_JSON}")
# How many classes?
# ids = {Path(x["path"]).parent.name for x in filtered}
# print("Unique IDs:", len(ids))

# # Confirm mapping correctness
# sample = filtered[0]["path"]
# rel = Path(sample).relative_to(SRC_PREFIX)
# print("Mapped path:", DST_ROOT / rel)
