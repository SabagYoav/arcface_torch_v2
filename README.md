# Knowledge Distillation for Gradually Masked Face Recognition

This repository investigates **partial face recognition** through knowledge distillation. The core research question: *How well can we recognize faces when only a partial facial region is visible?*

Student models are trained on progressively smaller **regions-of-interest (ROI)** centered around the eye area, while a full-face teacher model provides supervision via contrastive alignment. ROI ratios range from **15%** (eyes only) to **100%** (full face), enabling a systematic study of how facial coverage affects recognition accuracy.

Built on top of the [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) framework with distributed training, Partial FC, and mixed precision support.

---

## Overview

### Approach

1. **Teacher model**: A ResNet-50 trained on full-face images (Glint360K) using ArcFace loss.
2. **Student models**: Trained on cropped face regions at various ROI ratios, using one of several loss strategies.
3. **Knowledge distillation**: A CLIP-style contrastive loss aligns partial-face embeddings with full-face embeddings from the teacher.

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
├── train_v3_arcface.py          # ArcFace/CosFace training on ROI data
├── train_v4_clip.py             # CLIP-style knowledge distillation (partial → full)
├── train_v5_pairs.py            # Paired image training
├── train_v2_triplet.py          # Triplet loss training
├── training_multi_loops.py      # Orchestrates training across all ROI ratios
├── backbones/                   # ResNet, ViT, MobileFaceNet architectures
├── configs/                     # Experiment configurations
│   ├── base.py                  # Base config
│   └── variants_config_ratio_*.py  # Per-ratio configs (15, 20, 25, ..., 100)
├── eval/                        # Evaluation scripts
│   ├── roc_curve_multi.py       # ROC curves across multiple ratios
│   ├── roc_curve_single.py      # Single model verification
│   └── verification.py          # K-fold cross-validation
├── data_scratches/              # Dataset preparation (ROI extraction)
├── data_analytics/              # Pose estimation and analysis
├── losses.py                    # ArcFace, Triplet, CLIP contrastive losses
├── partial_fc_v2.py             # Partial FC for large-scale classification
└── utils/                       # Config loading, logging utilities
```

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/get-started/previous-versions/) >= 1.12.0
- `pip install -r requirement.txt`

---

## Training

### Standard ArcFace training on a specific ROI ratio

```shell
python train_v3_arcface.py configs/exp_glint360k_roi_*_r50_arcface.py
```

### Knowledge distillation (CLIP-style, partial → full face)

```shell
python train_v4_clip.py configs/glint360k_r50_clip
```

### Run all ROI ratio experiments sequentially

```shell
python training_multi_loops.py
```

---

## Dataset Preparation

This project uses [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) (360k IDs, 17.1M images) as the base dataset.

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

### Pose-stratified performance analysis

```shell
python data_analytics/face_pose_estimation.py
python eval/pose_peformance_eval_v3.py
```

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
