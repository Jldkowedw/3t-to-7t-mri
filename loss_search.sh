#!/usr/bin/env bash
set -euo pipefail

# ===============================================================
# run_stage3.sh   — Fine-tune Top-5 configs for 30 epochs each
# ===============================================================

# ───── Paths ─────
BASE="/home/yaxin/My_Files/3t-to-7t-mri/6loss_grid_random"
ROOT_DIR="/home/yaxin/My_Files/3t-to-7t-mri/Aligned"
TOP5="$BASE/top5_configs.csv"
OUT3="$BASE/stage3"
SUMMARY="$BASE/metrics_summary.csv"

mkdir -p "$OUT3"

# ───── Loop over Top-5 configs ─────
tail -n +2 "$TOP5" | while IFS=, read -r save_dir \
    lambda_l1 lambda_fm lambda_gan lambda_cycle lambda_percep lambda_gp \
    epochs lr batch_size noise_std avg_L1 avg_SSIM avg_PSNR; do

  name=$(basename "$save_dir")
  RUN_DIR="$OUT3/$name"
  echo "▶ Fine-tuning config: $name"
  echo "   λ_L1=$lambda_l1  λ_FM=$lambda_fm  λ_GAN=$lambda_gan"
  echo "   λ_CYCLE=$lambda_cycle  λ_PERCEP=$lambda_percep  λ_GP=$lambda_gp"
  echo

  python test.py \
    --mode train \
    --root_dir       "$ROOT_DIR" \
    --save_dir       "$RUN_DIR" \
    --epochs         30 \
    --lr             1e-4 \
    --batch_size     4 \
    --noise_std      0.01 \
    --slice_min      160 \
    --slice_max      260 \
    --lambda_l1      "$lambda_l1" \
    --lambda_fm      "$lambda_fm" \
    --lambda_gan     "$lambda_gan" \
    --lambda_cycle   "$lambda_cycle" \
    --lambda_percep  "$lambda_percep" \
    --lambda_gp      "$lambda_gp" \
    --summary_csv    "$SUMMARY"

  echo
done

echo "✅ Stage 3 fine-tuning complete. Results in $OUT3/"
