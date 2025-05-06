#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use('Agg')              # headless backend
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Update this to your actual path ───
BASE = Path("/home/yaxin/My_Files/3t-to-7t-mri/6loss_grid_random")
SUMMARY = BASE / "metrics_summary.csv"
STAGE3  = BASE / "stage3"

# Sanity checks
if not SUMMARY.exists():
    raise FileNotFoundError(f"Summary CSV not found: {SUMMARY}")
if not STAGE3.exists():
    raise FileNotFoundError(f"Stage3 directory not found: {STAGE3}")

# ── Stage 2: SSIM vs PSNR scatter ──
df = pd.read_csv(SUMMARY)
fig, ax = plt.subplots(figsize=(6, 4))
sizes = 100 * (1.0 / df["avg_L1"])
ax.scatter(df["avg_SSIM"], df["avg_PSNR"], s=sizes, alpha=0.7)
ax.set_xlabel("Validation SSIM")
ax.set_ylabel("Validation PSNR (dB)")
ax.set_title("Stage 2: Random Sweep\nSSIM vs PSNR")
ax.grid(True)
fig.tight_layout()
fig.savefig("stage2_scatter.png")
print("✅ Saved: stage2_scatter.png")

# ── Stage 3: SSIM convergence curves ──
fig2, ax2 = plt.subplots(figsize=(6, 4))
for run_dir in sorted(STAGE3.iterdir()):
    log_file = run_dir / "val_log.csv"
    if not log_file.exists():
        continue
    df_log = pd.read_csv(log_file)
    ax2.plot(df_log["epoch"], df_log["avg_PSNR"], label=run_dir.name)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation PSNR")
ax2.set_title("Stage 3: Fine-Tuning PSNR Curves")
ax2.legend(loc="lower right", fontsize="small", frameon=False)
fig2.tight_layout()
ax2.grid(True)
fig2.savefig("stage3_PSNR_curves.png")
print("✅ Saved: stage3_PSNR_curves.png")
