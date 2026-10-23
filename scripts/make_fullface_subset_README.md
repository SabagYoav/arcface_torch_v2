# Glint360K "3k" full-face subset

Documents exactly how `make_fullface_subset.py` builds the small, symlinked
identity subset used for quick experiments, so it can be reproduced
identically later.

## Source and destination

- Source: `/media/yoav/Yoav/datasets/glint360k/imageFolder_split_fullface`
  (already split into `train/`, `val/`, `test/` identity folders)
- Destination: `/media/yoav/Yoav/datasets/glint360k/fullface_subset`
  (mirrors the same three splits)

## Identity counts

| Split | Identities |
|---|---|
| train | 2500 |
| val | 500 |
| test | 50 |
| **Total** | **3050** |

("3k ids" refers to this 3,050-identity total — not the unrelated 36k/9k
`exp_arch_comparison_10perid` subset, and not the 63k–85k-identity pool
behind the `config_glint360k_subset_fullface` teacher checkpoint.)

## Selection method (deterministic, no seed needed)

For each split, independently:

1. List `SRC/<split>`'s entries via `os.scandir`, keep only directories whose
   name is all-digits (the identity-ID folders).
2. Sort those entries **by folder name, ascending** (`sorted(..., key=lambda e: e.name)`)
   — lexicographic string sort, not numeric (e.g. `"100"` sorts before `"2"`).
3. Walk the sorted list in order. For each identity folder, count its images
   (files with extension `.jpg`, `.jpeg`, or `.png`, case-insensitive). Skip
   the folder if that count is **< `MIN_IMAGES` = 3**.
4. Keep the **first N qualifying identities** in that sorted order (N = 2500
   for train, 500 for val, 50 for test). Stop as soon as N are picked — the
   rest of the sorted list is never examined.

Because this is a pure function of folder names + per-folder image counts,
re-running it against an unchanged `SRC` tree always picks the exact same
3,050 identities — no random seed is involved (unlike the 36k/9k subset,
which uses `random.Random(seed=42).shuffle(...)`).

## Build step

For each split:

1. If `DST/<split>` already exists, `shutil.rmtree` it first (full rebuild,
   not incremental).
2. Re-create `DST/<split>`.
3. For each of the N selected identities, `os.symlink(src_identity_dir,
   dst_split/<identity_name>)` — the **directory itself** is symlinked, not
   the individual images inside it. This makes the subset instant and
   disk-free; any downstream code that walks/globs the destination
   transparently follows the links back to the real images.

## To reproduce exactly

1. Ensure `SRC` (`/media/yoav/Yoav/datasets/glint360k/imageFolder_split_fullface`)
   is mounted and unchanged — the external drive this path lives on has been
   observed unmounted in some environments; mount it before running.
2. Run:
   ```bash
   python3 scripts/make_fullface_subset.py
   ```
3. Verify counts match the table above:
   ```bash
   for s in train val test; do echo "$s: $(ls /media/yoav/Yoav/datasets/glint360k/fullface_subset/$s | wc -l)"; done
   ```

## Caveats

- `SRC` must contain the exact same identity folders with the exact same
  image counts as when the original subset was built — adding, removing, or
  changing images in `SRC` (even ones not selected) can shift which
  identities fall above/below the `MIN_IMAGES` threshold earlier in the
  sorted order, changing the selected set.
- `MIN_IMAGES` only filters per-identity image **count**; it does not check
  whether faces are detectable in those images (no face-detection filtering
  happens here, unlike some other dataset-prep scripts in this repo).
