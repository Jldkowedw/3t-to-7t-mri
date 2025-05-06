# MRI Translation Hyperparameter Search Repository

This repository contains data, baseline code, improved model code, and a multi-stage hyperparameter search pipeline (random coarse sweep + fine-tuning) for a 3T→7T MRI translation task.

---

## Repository Layout

```
3t-to-7t-mri/                  # Project root
├─ Aligned/                     # Preprocessed input data (registered MRI volumes)
├─ baseline/                    # UNet+GAN baseline implementation and results
│   ├─ unet_baseline.py         # Baseline train/test script
│   └─ unet_baseline/           # Baseline checkpoints & test metrics
├─ Final_code/                  # Improved model implementation and final results
│   ├─ final_model.py           # Improved model train/test script
│   └─ final_model/             # Fine-tuned improved model checkpoints & metrics
├─ grid_search/                 # Hyperparameter search pipeline
│   ├─ 6loss_grid_random/       # Stage 2 outputs (random 6-loss sweep)
│   │   ├─ metrics_summary.csv   # Summary of 50 random configs (3 epochs each)
│   │   ├─ stage2_scatter.png    # SSIM vs PSNR scatter plot
│   │   └─ stage3/               # Stage 3 fine-tuning outputs (top-3)
│   │       ├─ <config_name>/    # Each top-3 config folder
│   │       │   └─ val_log.csv   # Epoch-wise validation metrics log
│   │       ├─ stage3_ssim_curves.png  # Validation SSIM convergence plot
│   │       └─ stage3_psnr_curves.png  # Validation PSNR convergence plot
│   ├─ run_single_job.sh        # Stage 2 driver: 50 random combos (3 epochs)
│   ├─ graph.py                 # Plots Stage 2 scatter from metrics_summary.csv
│   ├─ loss_search.sh           # Stage 3 driver: fine-tune top-3 (30 epochs)
│   ├─ final_graph.py           # Aggregates and plots Stage 2 & Stage 3 results
│   └─ test.py                  # Core train/test script with --summary_csv support
```

---

## Description of Core Scripts

### `unet_baseline.py`

Baseline UNet+GAN training and testing. Outputs model checkpoints and test metrics under `baseline/unet_baseline/`.

### `final_model.py`

Improved model training/testing script, functionally identical to `test.py` but used to finalize and export the best model for deployment.

### `run_single_job.sh`

Stage 2 driver: samples 50 random hyperparameter combinations of the six loss weights, trains each for 3 epochs via `test.py`, and writes `metrics_summary.csv` to `6loss_grid_random/`.

### `graph.py`

Reads `6loss_grid_random/metrics_summary.csv` and generates a validation SSIM vs PSNR scatter plot (`stage2_scatter.png`).

### `loss_search.sh`

Stage 3 driver: reads a CSV of the top-3 configurations, fine-tunes each for 30 epochs via `test.py`, and logs per-epoch metrics in `6loss_grid_random/stage3/<config>/val_log.csv`.

### `final_graph.py`

Combines Stage 2 and Stage 3 logs to produce:

* `stage2_scatter.png` (Stage 2 scatter plot)
* `stage3_ssim_curves.png` (SSIM convergence)
* `stage3_psnr_curves.png` (PSNR convergence)

### `test.py`

Unified training/testing script supporting:

* `--mode {train,test}`
* Dataset paths & slice ranges
* Noise standard deviation
* Loss weight parameters (`--lambda_l1`, `--lambda_fm`, `--lambda_gan`, `--lambda_cycle`, `--lambda_percep`, `--lambda_gp`)
* `--summary_csv` to append average test metrics to a global CSV

---

## Quickstart

python final_model.py --mode train
python final_model.py --mode test

---

## Contact

Yaxin Su (yaxin.su@yale.edu)

