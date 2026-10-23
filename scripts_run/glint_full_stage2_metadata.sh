#!/bin/bash
# Stage 2 of glint_full_finetune_pipeline.sh (runs INSIDE the devcontainer):
# rebuild /DATA/glint360k_download/eye_center_metadata/center_eyes_metadata.json
# for the full glint360k with data_scratches/build_eye_center_metadata.py.
set -euo pipefail
cd /home/yoav/arcface_torch_v2

OUT=/DATA/glint360k_download/eye_center_metadata
mkdir -p "$OUT"

# Seed the builder's resume files with the eye centers already computed for the
# 36k/9k arch-comparison identities (same detector + images), so it skips them.
if [ ! -f "$OUT/.seeded" ]; then
    python3 - <<'EOF'
import json
NUM_SHARDS = 4
OUT = "/DATA/glint360k_download/eye_center_metadata"
meta = {}
for split in ("train", "val"):
    meta.update(json.load(open(f"/DATA/glint360k_arch_comparison/{split}_eye_center_metadata.json")))
by_shard = {s: {} for s in range(NUM_SHARDS)}
for k, v in meta.items():
    id_name = k.split("/", 1)[0]
    by_shard[int(id_name) % NUM_SHARDS].setdefault(id_name, []).append((k, v))
for s, ids in by_shard.items():
    with open(f"{OUT}/shard_{s}.jsonl", "a") as jf, open(f"{OUT}/shard_{s}.done_ids.txt", "a") as df:
        for id_name, recs in ids.items():
            for k, v in recs:
                jf.write(json.dumps({"k": k, "center_y": v["center_y"], "detected": v["detected"]}) + "\n")
            df.write(id_name + "\n")
print(f"seeded {len(meta)} entries from the 45k arch-comparison identities")
EOF
    touch "$OUT/.seeded"
fi

# onnxruntime's CUDA EP needs cuDNN 9 / cuBLAS from torch's bundled nvidia wheels.
export LD_LIBRARY_PATH=$(ls -d /home/yoav/.local/lib/python3.10/site-packages/nvidia/*/lib | grep -v cu13 | tr "\n" ":")${LD_LIBRARY_PATH:-}
python3 data_scratches/build_eye_center_metadata.py   # 4 shards (1/GPU), merges at the end

python3 - <<'EOF'
import json
m = json.load(open("/DATA/glint360k_download/eye_center_metadata/center_eyes_metadata.json"))
det = sum(v["detected"] for v in m.values())
ids = len({k.split("/", 1)[0] for k, v in m.items() if v["detected"]})
print(f"metadata check: {len(m)} entries, {det} detected, {ids} identities with detections")
assert len(m) > 17_000_000, "metadata looks incomplete"
EOF
