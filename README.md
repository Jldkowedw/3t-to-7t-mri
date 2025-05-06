# MRI Translation Hyperparameter Search Repository

This repository supports research in **3T-to-7T brain MRI image translation**. It contains preprocessing, baseline and improved model training scripts, and a two-stage hyperparameter search pipeline for tuning six loss components in a GAN-based architecture.

---

## 📁 Repository Layout

3t-to-7t-mri/
├── Aligned/ # Preprocessed paired 3T/7T MRI slices
├── baseline.py # Baseline UNet+GAN training/testing script
├── final_model.py # Improved model script (finalized for deployment)
├── test.py # Core training/testing script (used by search scripts)
├── run_single_job.sh # Stage 2: 50-random-combo 3-epoch grid search
├── loss_search.sh # Stage 3: fine-tuning top configs for 30 epochs
├── graph.py # Stage 2: SSIM vs PSNR scatter plot
├── final_graph.py # Stage 2 + Stage 3 plot aggregator
├── README.md # This file

Always show details


---

## 🔧 Core Scripts

### `baseline.py`
Trains a UNet+GAN baseline model. Outputs model checkpoints and test metrics to the `baseline/` directory.

### `final_model.py`
Trains and tests the improved model with optimal hyperparameters. Used for final evaluation and export.

### `test.py`
General-purpose script used by both search stages. Accepts:
- `--mode {train,test}`
- All six loss weight arguments:  
  `--lambda_l1`, `--lambda_fm`, `--lambda_gan`, `--lambda_cycle`, `--lambda_percep`, `--lambda_gp`
- `--summary_csv`: logs average PSNR/SSIM results to a global CSV.

---

## 🔍 Grid Search Pipeline

### **Stage 2: Random Sweep**
**Script**: `run_single_job.sh`  
- Samples 50 random 6-loss configurations.
- Trains each for 3 epochs using `test.py`.
- Saves to: `grid_search/6loss_grid_random/metrics_summary.csv`
- Output:  
  - `stage2_scatter.png`: SSIM vs PSNR plot

### **Stage 3: Fine-Tuning**
**Script**: `loss_search.sh`  
- Selects top 3 configs from Stage 2.
- Fine-tunes each for 30 epochs.
- Saves per-epoch logs under: `stage3/<config_name>/val_log.csv`
- Output:  
  - `stage3_ssim_curves.png`
  - `stage3_psnr_curves.png`

---

## 📊 Visualization

### `graph.py`
Reads `metrics_summary.csv` and creates a scatter plot of Stage 2 (SSIM vs PSNR).

### `final_graph.py`
Aggregates Stage 2 & Stage 3 results:
- `stage2_scatter.png`
- `stage3_ssim_curves.png`
- `stage3_psnr_curves.png`

---

## 🚀 Quickstart

**Train final model:**
bash
python final_model.py --mode train
Test final model:

Always show details

python final_model.py --mode test
📬 Contact

Maintainer: Yaxin Su
Email: yaxin.su@yale.edu
