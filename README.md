# MRI Translation Hyperparameter Search Repository

This repository contains all scripts, outputs, and instructions for a multi-stage hyperparameter search and visualization pipeline applied to a 3T→7T MRI translation task.

---

## Directory Structure

```
Aligned/                       # Input data (registered 3T and 7T volumes)
baseline/                      # UNet+GAN baseline implementation and output
├─ unet_baseline/              # Baseline model checkpoints & test metrics
├─ unet_baseline.py            # Baseline training/testing script

Final_code/                    # Improved model implementation and final results
├─ final_model/                # Fine-tuned improved model checkpoints & metrics
├─ final_model.py              # Improved model training/testing script

grid_search/                   # Multi‑stage hyperparameter search
├─ 6loss_grid_random/          # Stage 2: random coarse sweep outputs
│    ├─ metrics_summary.csv     # Summary of 50 random combos (3 epochs each)
│    ├─ stage2_scatter.png      # SSIM vs PSNR scatter plot
│    └─ stage3/                 # Stage 3: fine‑tuning outputs for top‑3
│         ├─ <config_name>/     # Each top‑3 config’s folder
│         │    └─ val_log.csv   # Epoch‑wise validation metrics log
│         ├─ stage3_ssim_curves.png  # Validation SSIM curves
│         └─ stage3_psnr_curves.png  # Validation PSNR curves
|
├─ graph.py                    # Generates SSIM vs PSNR scatter plot
├─ run_single_job.sh           # Stage 2 driver: random sweep (50 combos)
├─ loss_search.sh              # Stage 3 driver: fine‑tune top‑3 (30 epochs)
├─ final_graph.py              # Aggregates and plots Stage 2 & 3 results
├─ test.py                     # Core train/test script with --summary_csv support
```

grid\_search/                      # Main grid search folder
├─ 6loss\_grid\_random/             # Stage 2: random coarse sweep outputs
│    ├─ metrics\_summary.csv        # Summary of 50 random combos (3 epochs each)
│    ├─ stage2\_scatter.png         # SSIM vs PSNR scatter plot
│    └─ stage3/                    # Stage 3: fine-tuning outputs
│         ├─ \<config\_name>/        # Each top-3 config’s folder
│         │    └─ val\_log.csv      # Epoch-wise validation metrics log
│         ├─ stage3\_ssim\_curves.png  # Validation SSIM curves for top-3
│         └─ stage3\_psnr\_curves.png  # Validation PSNR curves for top-3
|
├─ graph.py                       # Generates Stage 2 scatter plot from metrics\_summary.csv
├─ final\_graph.py                 # Builds Stage 2 & Stage 3 plots from CSV/log files
├─ test.py                        # Core train/test script with --summary\_csv support

loss\_search.sh                    # Stage 3 fine-tuning script for top-3 configs (30 epochs)
run\_single\_job.sh                 # Stage 2 random sweep driver (50 combos, 3 epochs)

````

---

## Script Descriptions

### `graph.py`
- **Purpose:** Reads `metrics_summary.csv` (Stage 2), creates an SSIM vs PSNR scatter plot (`stage2_scatter.png`).
- **Input:** `6loss_grid_random/metrics_summary.csv`
- **Output:** `6loss_grid_random/stage2_scatter.png`

### `run_single_job.sh`
- **Purpose:** Randomly samples 50 hyperparameter combinations and invokes `test.py` for each (3 epochs each), producing `metrics_summary.csv`.

### `loss_search.sh`
- **Purpose:** Reads the `top3_configs.csv` file listing the three best combos, fine-tunes each for 30 epochs using `test.py`, and appends their results to `metrics_summary.csv`.
- **Output:** `6loss_grid_random/stage3/<config_name>/val_log.csv` and updated `metrics_summary.csv`

### `final_graph.py`
- **Purpose:** Aggregates Stage 2 (`metrics_summary.csv`) and Stage 3 (`val_log.csv`) data, then generates:
  - `stage2_scatter.png` (Stage 2 scatter)
  - `stage3_ssim_curves.png` and `stage3_psnr_curves.png` (Stage 3 convergence curves)

### `test.py`
- **Purpose:** Training & testing engine. Supports:
  - `--mode {train,test}`
  - `--summary_csv` to append average metrics to a global CSV
  - Loss weight parameters (`--lambda_l1`, `--lambda_fm`, `--lambda_gan`, `--lambda_cycle`, `--lambda_percep`, `--lambda_gp`)

---

## Usage Example

```bash
# 1) Stage 2: Random Coarse Sweep
bash run_single_job.sh         # runs 50 combos × 3 epochs
python graph.py               # plots SSIM vs PSNR

# 2) Stage 3: Fine-Tuning Top 3
bash loss_search.sh           # fine-tunes each top-3 for 30 epochs

# 3) Visualization: Final Graphs
python final_graph.py         # generates all final plots
````

---

## Contact

Yaxin Su (yaxin.su@yale.edu)

