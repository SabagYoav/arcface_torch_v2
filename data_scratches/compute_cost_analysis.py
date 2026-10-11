"""Computational cost analysis: params, FLOPs, and per-image inference latency
for the teacher (R50) and student (Swin-Base, best_model from clip_ratio_15).
Relevant for latency-sensitive surveillance deployment.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from ptflops import get_model_complexity_info

from backbones import get_model

DEVICE = "cuda"
INPUT_RES = (3, 112, 112)
N_WARMUP = 30
N_ITERS = 300

TEACHER_CKPT = "16backbone.pth"
STUDENT_CKPT = "work_dirs/exp_clip_r50_vs_swin_glint/clip_ratio_15/best_model.pt"


def load_model(name, ckpt_path):
    m = get_model(name, dropout=0.0, fp16=True, num_features=512).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE)
    m.load_state_dict(sd)
    m.eval()
    return m


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def measure_flops(m):
    macs, params = get_model_complexity_info(
        m, INPUT_RES, as_strings=False, print_per_layer_stat=False, verbose=False,
    )
    return macs, params


@torch.no_grad()
def measure_latency(m, batch_size=1):
    x = torch.randn(batch_size, *INPUT_RES, device=DEVICE)
    # warmup
    for _ in range(N_WARMUP):
        with torch.cuda.amp.autocast(True):
            m(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(True):
            m(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    times.sort()
    n = len(times)
    mean_ms = sum(times) / n
    median_ms = times[n // 2]
    p95_ms = times[int(n * 0.95)]
    return mean_ms, median_ms, p95_ms


def analyze(name, arch_name, ckpt_path):
    print(f"\n{'='*60}\n{name} ({arch_name})\n{'='*60}")
    m = load_model(arch_name, ckpt_path)

    n_params = count_params(m)
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f} M)")

    macs, _ = measure_flops(m)
    flops = macs * 2
    print(f"MACs:  {macs:,} ({macs/1e9:.3f} GMac)")
    print(f"FLOPs: {flops:,} ({flops/1e9:.3f} GFLOPs)")

    mean_ms, median_ms, p95_ms = measure_latency(m, batch_size=1)
    print(f"Inference latency (batch=1, fp16, GPU): "
          f"mean={mean_ms:.3f} ms  median={median_ms:.3f} ms  p95={p95_ms:.3f} ms")
    print(f"Throughput (batch=1): {1000.0/mean_ms:.1f} images/sec")

    return {
        "name": name, "arch": arch_name, "params": n_params, "macs": macs, "flops": flops,
        "latency_mean_ms": mean_ms, "latency_median_ms": median_ms, "latency_p95_ms": p95_ms,
    }


def main():
    results = []
    results.append(analyze("Teacher", "r50", TEACHER_CKPT))
    results.append(analyze("Student", "swin_base", STUDENT_CKPT))

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    hdr = f"{'Model':<10}{'Arch':<12}{'Params (M)':<12}{'GFLOPs':<10}{'Latency (ms)':<14}{'Speedup':<10}"
    print(hdr)
    teacher_lat = results[0]["latency_mean_ms"]
    for r in results:
        speedup = teacher_lat / r["latency_mean_ms"]
        print(f"{r['name']:<10}{r['arch']:<12}{r['params']/1e6:<12.2f}{r['flops']/1e9:<10.3f}"
              f"{r['latency_mean_ms']:<14.3f}{speedup:<10.2f}x")


if __name__ == "__main__":
    main()
