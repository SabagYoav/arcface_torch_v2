import sys
import os
import torch
import numpy as np

os.sys.path.append('.')

from eval.kpis.evaluation_ver import FaceVerificator, init_params as ver_init_params
from eval.kpis.evaluation_id import FaceIditificator, init_params as id_init_params


CONFIG_FILES = [
    # "configs/exp_glint360k_roi_15_r50_arcface.py",
    # "configs/exp_glint360k_roi_20_r50_arcface.py",
    # "configs/exp_glint360k_roi_60_r50_arcface.py",
    "configs/exp_glint360k_roi_100_r50_cosface.py"

]

THRESHS = [round(t, 2) for t in np.arange(0.10, 0.41, 0.01).tolist()]


def run_all(config_files):
    results = []

    for cfg_path in config_files:
        print(f"\n{'='*60}")
        print(f"Config: {cfg_path}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        np.random.seed(42)

        # --- shared init (loads cfg, dataloader, backbone) ---
        # Both scripts use the same init logic; use ver_init_params once
        try:
            cfg, dataloader, backbone = ver_init_params(cfg_path)
        except Exception as e:
            print(f"  SKIP  {cfg_path}: {e}")
            continue

        # --- Verification ---
        print("\n--- Verification ---")
        ver_evaluator = FaceVerificator(subsample=False)
        ver_evaluator.plt_save_path = f"{cfg.output}/ver_scores_distributions_plot.png"
        TP, TN, FP, FN = ver_evaluator.compute_verification_acc(
            backbone, dataloader=dataloader, threshs=THRESHS, device='cuda'
        )
        total = TP + TN + FP + FN
        acc = (TP + TN) / total if total else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        print(f"dataset: {cfg.test_data_root} | Accuracy: {acc:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")

        # --- Identification ---
        print("\n--- Identification ---")
        id_evaluator = FaceIditificator(max_classes=1000, max_instances=None)
        id_results = id_evaluator.compute_retrieval_ranks_with_knn(
            backbone, dataloader=dataloader, device='cuda'
        )
        print(id_results)

        results.append({
            "config": cfg_path,
            "test_data": cfg.test_data_root,
            "ver_acc": acc,
            "ver_recall": recall,
            "ver_precision": precision,
            **id_results,
        })

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['config']:55s} | acc={r['ver_acc']:.4f} | recall={r['ver_recall']:.4f} | prec={r['ver_precision']:.4f} | rank@1={r.get('rank@1',0):.2f}% | rank@5={r.get('rank@5',0):.2f}%")


if __name__ == "__main__":
    config_files = sys.argv[1:] if len(sys.argv) > 1 else CONFIG_FILES
    run_all(config_files)
