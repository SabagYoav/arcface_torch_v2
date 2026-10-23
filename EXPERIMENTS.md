# Experiment Log — Gradually Masked Face Recognition (Knowledge Distillation)

Teachers:
- **subset teacher** = `work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt` (R50, trained on Glint360K subset)
- **16backbone** = `work_dirs/16backbone.pth` (full-Glint360K R50 = the InsightFace baseline; the Swin student was distilled against THIS)

| # | Date | Experiment | Student → Teacher | Dataset | ROI | Protocol / Metrics | Output |
|---|------|-----------|-------------------|---------|-----|--------------------|--------|
| A | ~Jun | CLIP distillation sweep | R50 → R50 (subset) | Glint360K subset (~2.5k ids) + variants_dataset | 1.0,0.6,0.4,0.3,0.2,0.15 | val balanced-acc vs ROI | work_dirs/exp_clip_r50_vs_r50/ |
| B | ~Jun | CLIP distillation sweep | ViT-S → R50 (subset) | same as A | 1.0…0.15 | val balanced-acc vs ROI | work_dirs/exp_clip_r50_vs_vit/ |
| C | Jul 15 | Standard benchmark (3-mode) | ViT-S → R50 (subset) | LFW,CALFW,CPLFW,AgeDB-30,CFP-FF/FP | 20% | 10-fold acc, TAR@FAR, AUC; Full/Cross/Student | standard_bench_vit_ratio20.json |
| D | Jul 20 | Reference baselines (3-mode) | InsightFace R50, ElasticFace R100, MaskInv-HG/LG R100 | same 6 benchmarks | 20% | 3-mode protocol | reference_three_modes.json |
| E | Jul 30 | Open-set identification | ViT-S → R50 (subset) | Glint360K fullface + variants_dataset/ratio_20 | 20% | 3k ids; Rank@1/5/10, FAR, DIR@rank1 | open_set_identification.json |
| F | Aug 3 | Standard benchmark (3-mode) | Swin-B → R50 (16backbone) | same 6 benchmarks | 15% | 3-mode, 10-fold | swin_ratio15_three_modes.json |
| G | Aug 4 | RFW benchmark (3-mode, per race) | Swin-B + 4 baselines → R50 (16backbone) | RFW (Caucasian/African/Asian/Indian) | 15% band | 3-mode per race | rfw_ratio15_three_modes.json |
| H | Aug 21 | Open-set identification | Swin-B → R50 (16backbone) | Glint360K fullface + variants_dataset/ratio_15 | 15% | Rank@1/5/10, FAR, DIR@rank1 | open_set_identification_swin15.json |

## Headline results
- A (R50 sweep, val acc): 100%→95.0, 60%→92.8, 40%→89.8, 30%→86.3, 20%→80.4, 15%→71.9
- B (ViT-S sweep, val acc): 100%→93.0, 60%→89.4, 40%→83.2, 30%→80.3, 20%→70.4, 15%→67.0
- C (ViT-S 20%, Partial→Full acc): LFW 97.8 | CALFW 91.8 | CPLFW 76.4 | AgeDB 88.7 | CFP-FF 97.3 | CFP-FP 80.3
- D (baselines): all strong; MaskInv most partial-robust (partial↔partial ~94–98)
- E (open-set, ViT-S 20%): Rank@1 97.2 / @5 99.2 / @10 99.5; DIR@rank1 81.0 @FAR1%, 75.2 @FAR0.1%
- F (Swin 15%, Partial→Full acc): LFW 99.1 | CALFW 94.5 | CPLFW 89.7 | AgeDB 94.0 | CFP-FF 99.2 | CFP-FP 96.6
- G (RFW, Swin 15% Partial→Full): Caucasian 93.8 | African 88.7 | Asian 88.1 | Indian 87.9 (baselines ~95–98)
- H (open-set, Swin 15%): Rank@1 90.2 / @5 96.2 / @10 97.1; DIR@rank1 73.5 @FAR1%, 60.3 @FAR0.1%

## Report revision mapping (best inference — confirm)
- Only `Final_Report_v6.1.pdf` (~Jul 10) seen. Its Table VI == experiment C (ViT-S 20%); ROI-sweep results == A/B.
- D, E, F, G, H are POST-v6.1 (not yet in report) — candidates for the next revision.

## Caveats
- Teacher switch: C/E use the SUBSET teacher; F/G/H use the full-Glint 16backbone teacher → not strictly cross-comparable.
- ROI labels 15% / 20% are the true content fractions (earlier "~45%" was a measurement artifact from `any()`-based band detection).
- Result JSONs live in `work_dirs/benchmarks/`.
