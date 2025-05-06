# MRI Translation Hyperparameter Search Repository



This repository supports research in **3T-to-7T brain MRI image translation**. It contains preprocessing, baseline and improved model training scripts, and a two-stage hyperparameter search pipeline for fine-tuning six loss terms in a GAN-based architecture.

## 📁 Repository Layout

3t-to-7t-mri/
├── Aligned/ # Preprocessed paired 3T/7T data
├── baseline.py # Baseline UNet+GAN training/testing
├── final_model.py # Improved model for training/testing
├── test.py # Shared training/test script for grid search
├── run_single_job.sh # Stage 2: random 6-loss sweep (3 epochs)
├── loss_search.sh # Stage 3: fine-tuning top configs (30 epochs)
├── graph.py # Plots Stage 2 SSIM vs PSNR scatter
├── final_graph.py # Aggregates Stage 2 & 3 plots
├── README.md # You are here!


## 🔧 Core Scripts

### `baseline.py`
Runs baseline UNet+GAN training and evaluation. Results stored under `baseline/`.

### `final_model.py`
Trains the improved model using best-found hyperparameters. Used to export final deployable results.

### `test.py`
Shared script used by both grid search stages. Accepts all key parameters:
```bash
--mode {train,test}
--lambda_l1, --lambda_fm, --lambda_gan, --lambda_cycle, --lambda_percep, --lambda_gp
--summary_csv                # logs average metrics
🔍 Grid Search Pipeline

Stage 2: Random Sweep (run_single_job.sh)
Samples 50 random 6-loss combinations.
Runs 3-epoch trials via test.py.
Saves metrics in grid_search/6loss_grid_random/metrics_summary.csv.
Outputs:
stage2_scatter.png (SSIM vs PSNR)
Stage 3: Fine-Tuning (loss_search.sh)
Selects top-3 from Stage 2.
Fine-tunes each for 30 epochs.
Logs SSIM and PSNR curves:
stage3_ssim_curves.png
stage3_psnr_curves.png
📊 Visualization

graph.py
Creates Stage 2 scatter plot (stage2_scatter.png).

final_graph.py
Combines and plots:

Stage 2 scatter
Stage 3 SSIM/PSNR convergence curves
🚀 Quickstart

Train final model:
python final_model.py --mode train
Test final model:
python final_model.py --mode test
📬 Contact

Maintainer: Yaxin Su
📧 yaxin.su@yale.edu

