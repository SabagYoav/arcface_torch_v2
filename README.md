# Knowledge Distillation for Gradually Masked Face Recognition

This repository investigates **partial face recognition** through knowledge distillation. The core research question: *How well can we recognize faces when only a partial facial region is visible?*

Student models are trained on progressively smaller **regions-of-interest (ROI)** centered around the eye area, while a full-face teacher model provides supervision via contrastive alignment. ROI ratios range from **15%** (eyes only) to **100%** (full face), enabling a systematic study of how facial coverage affects recognition accuracy.

Built on top of the [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) framework with distributed training, Partial FC, and mixed precision support.

---

## Overview

### Approach

1. **Teacher model**: A ResNet-50 trained on full-face images (Glint360K) using ArcFace loss.
2. **Student models**: Trained on cropped face regions at various ROI ratios. Two student backbones are supported — **ResNet-50** and **ViT-S** — enabling an architecture comparison under identical distillation settings.
3. **Knowledge distillation**: A CLIP-style contrastive loss aligns partial-face (student) embeddings with full-face (teacher) embeddings.

### ROI Ratio Concept

The **ROI ratio** controls how much of the face is visible to the student model. Crops are centered at the eye midpoint, with height as a fraction of the total face bounding box:

| Ratio | Visible Region |
|-------|---------------|
| 15%   | Narrow Eyes only |
| 20-30% | Eyes only |
| 40-60% | Eyes and nose |
| 100%  | Full face |

ROI crops are placed on a black canvas maintaining the original 112x112 image dimensions.

---

## Project Structure

```
├── train_v4_clip.py             # CLIP-style knowledge distillation (partial → full)
├── train_v3_arcface.py          # ArcFace/CosFace training on ROI data
├── train_v2_triplet.py          # Triplet loss training
├── train_v5_pairs.py            # Paired image training
├── training_multi_loops.py      # Distillation sweep across all ROI ratios
│                                #   (build ROI data → train → verify → plot; resumable per-ratio)
├── backbones/                   # ResNet (r50), ViT-S, MobileFaceNet architectures
├── configs/
│   ├── base.py                          # Base config (defaults)
│   ├── experiment_r50_vs_r50_clip.py    # R50 teacher → R50  student sweep
│   ├── experiment_r50_vs_vit_clip.py    # R50 teacher → ViT-S student sweep
│   └── variants_config_ratio_*.py       # Auto-generated per-ratio configs
├── eval/
│   ├── standard_verification.py         # Cross-modal verification on the Glint360K val subset
│   └── benchmarks/
│       ├── standard_bench_all.py        # Standard 10-fold protocol: LFW/CALFW/CPLFW/AgeDB-30/CFP
│       ├── preprocess_lfw_partial.py    # Build aligned full + partial-face LFW crops
│       └── preprocess_cfp.py            # Build CFP (FF/FP) full + partial crops + pair lists
├── losses.py                    # ArcFace, Triplet, CLIP contrastive losses
├── partial_fc_v2.py             # Partial FC for large-scale classification
├── utils/                       # Config loading, logging, CLIP verification
└── work_dirs/                   # Checkpoints, plots, and benchmark results (see below)
```

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/get-started/previous-versions/) >= 1.12.0
- `pip install -r requirements.txt`
- `timm` (required for the ViT-S student), `insightface` (only for building benchmark partial crops)

---

## Training

Distillation experiments are driven by an **experiment config** + the **sweep orchestrator**
(`training_multi_loops.py`). For each ROI ratio the orchestrator builds the ROI-cropped
dataset, trains the student against the frozen full-face teacher (CLIP-style contrastive loss),
runs verification every epoch, and plots accuracy vs ROI. It is **resumable**: each finished
ratio writes a `result.json` marker and is skipped on the next launch.

Select the experiment via two settings at the top of `training_multi_loops.py`:
- `BASE_CONFIG` — student architecture + hyperparameters
  (`configs/experiment_r50_vs_r50_clip.py` or `configs/experiment_r50_vs_vit_clip.py`)
- `EXP_NAME` — output namespace (`work_dirs/<EXP_NAME>/`); can be overridden by env var.

### Run a distillation sweep across all ROI ratios `[1.0, 0.6, 0.4, 0.3, 0.2, 0.15]`

```shell
# R50 teacher -> R50 student
EXP_NAME=exp_clip_r50_vs_r50 python training_multi_loops.py    # BASE_CONFIG -> experiment_r50_vs_r50_clip.py

# R50 teacher -> ViT-S student
EXP_NAME=exp_clip_r50_vs_vit python training_multi_loops.py    # BASE_CONFIG -> experiment_r50_vs_vit_clip.py
```

The two experiment configs share every hyperparameter except the student `network`, so the
R50 and ViT-S runs are directly comparable.

### Train a single ROI ratio directly

```shell
python train_v4_clip.py --config configs/experiment_r50_vs_vit_clip.py
```

---

## Dataset Preparation

This project uses [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) (360k IDs, 17.1M images) as the base dataset.

### Download (training + test sets)

The prepared full-face and ROI datasets (both **training** and **test** splits) are available here:

**📦 [Datasets — Google Drive](https://drive.google.com/drive/folders/1RHBcO5BHr7U15Ua96SFsJtlOK1fZ7fqF?usp=sharing)**

Download and place them under the dataset root expected by the configs (e.g. `/media/<user>/.../datasets/`),
matching the `root_ff` / `root_pf` paths in the experiment configs.

### Generate ROI datasets at different ratios

```shell
python data_scratches/build_roi_datasets.py
```

This creates cropped datasets at `/datasets/glint360k/ROIs/ratio_{15,20,25,30,40,60,100}/{train,val,test}`.

---

## Evaluation

### ROC curves across all ROI ratios

```shell
python eval/roc_curve_multi.py
```

Compares partial-face models (15%–100%) against the full-face teacher, generating a combined ROC plot.

### Single model verification

```shell
python eval/roc_curve_single.py
```

### Standard benchmark verification (LFW / CALFW / CPLFW / AgeDB-30 / CFP)

Evaluates a trained student on standard face-recognition benchmarks under the **standard
10-fold verification protocol** on the official pair lists (threshold fit on train folds, no
leakage). Three settings are reported per benchmark: **teacher** full↔full, **cross-domain**
partial→full (student partial vs teacher full), and **student** partial↔partial.

```shell
python eval/benchmarks/standard_bench_all.py \
    --student work_dirs/exp_clip_r50_vs_vit/clip_ratio_20/best_model.pt \
    --student-net vit_s_dp005_mask_0 --ratio 20
```

Reports accuracy (10-fold mean ± std), TAR@FAR (1e-1 / 1e-2 / 1e-3), and AUC; full results
are written to `work_dirs/benchmarks/standard_bench_vit_ratio20.json`.

### Pose-stratified performance analysis

```shell
python data_analytics/face_pose_estimation.py
python eval/pose_peformance_eval_v3.py
```

---

## Results

### Standard benchmark verification

Verification accuracy (%) on standard face-recognition benchmarks, **standard 10-fold protocol**
(ViT-S `ratio_20` student, R50 teacher). **Teacher** = full↔full, **Cross-Domain** = partial→full
(student partial vs teacher full), **Student** = partial↔partial.

| Benchmark | Focus | Teacher (full↔full) | Cross-Domain (partial→full) | Student (partial↔partial) |
|-----------|-------|:---:|:---:|:---:|
| LFW       | Baseline        | 99.58 | 97.82 | 95.43 |
| CALFW     | Age             | 95.33 | 91.82 | 87.47 |
| CPLFW     | Pose            | 86.18 | 76.37 | 68.23 |
| AgeDB-30  | Age (30 years)  | 96.23 | 88.67 | 83.03 |
| CFP-FF    | Frontal–frontal | 99.73 | 97.27 | 95.31 |
| CFP-FP    | Frontal–profile | 91.36 | 80.34 | 69.03 |

Cross-domain alignment (partial→full) consistently outperforms partial↔partial, confirming that
anchoring partial-face embeddings to the full-face teacher space improves robustness under
reduced facial information. Frontal benchmarks stay near-ceiling (~97%); pose (CPLFW, CFP-FP)
is the hardest condition. TAR@FAR and AUC for every entry are in
`work_dirs/benchmarks/standard_bench_vit_ratio20.json`.



### Checkpoints

All trained checkpoints (teacher + R50 / ViT-S students across ROI ratios) are available here:

**🧠 [Checkpoints — Google Drive](https://drive.google.com/drive/folders/12NR3TcoKfjluAcgfHmrg-rIEqe5FY0oI?usp=drive_link)**

Download and place them with respect to the config you are running

---

## Acknowledgements

Based on the [ArcFace-Torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) implementation from InsightFace.

## Citations

```
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}
@inproceedings{an2022partialfc,
  author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
  title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
  booktitle={CVPR},
  year={2022},
}
```
