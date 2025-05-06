import os
import argparse
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# -------------------------------------------------------------------
# Default config (can be overridden via CLI)
# -------------------------------------------------------------------
ROOT_DIR   = "/home/yaxin/My_Files/3t-to-7t-mri/Aligned"
SAVE_DIR   = "/home/yaxin/My_Files/3t-to-7t-mri/unet_baseline"
SLICE_RANGE = (160, 260)
EPOCHS     = 30
BATCH_SIZE = 4
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# Dataset – Paired 3T & 7T slices
# -------------------------------------------------------------------
class PairedMRIDataset(Dataset):
    """Returns (3T slice, 7T slice, slice‑index). Slices normalised to [0,1]."""

    def __init__(self, root_dir: str, mode: str = "train"):
        super().__init__()
        subjects = [f"sub-{i:02d}" for i in range(1, 10)] if mode == "train" else ["sub-10"]
        self.samples = []
        for sid in subjects:
            p3 = os.path.join(root_dir, f"{sid}_ses-1_T1w_defaced_registered.nii")
            p7 = os.path.join(root_dir, f"{sid}_ses-2_T1w_defaced_registered.nii")
            if not (os.path.exists(p3) and os.path.exists(p7)):
                continue
            vol3 = nib.load(p3).get_fdata(); vol7 = nib.load(p7).get_fdata()
            if vol3.shape != vol7.shape:
                continue
            for idx in range(*SLICE_RANGE):
                s3, s7 = vol3[:, :, idx], vol7[:, :, idx]
                s3 = (s3 - s3.min()) / (s3.max() - s3.min() + 1e-8)
                s7 = (s7 - s7.min()) / (s7.max() - s7.min() + 1e-8)
                self.samples.append((s3, s7, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s3, s7, sl_idx = self.samples[idx]
        return (
            torch.tensor(s3[None], dtype=torch.float32),
            torch.tensor(s7[None], dtype=torch.float32),
            sl_idx,
        )

# -------------------------------------------------------------------
# Model – Lightweight UNet++ (single‑channel)
# -------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2), 
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNetPlusPlus(nn.Module):
    """4‑level UNet++，去掉最深一层 & 部分嵌套分支。参数量约 1/3。"""
    def __init__(self):
        super().__init__()
        # encoder 0‑3
        self.conv00 = DoubleConv(1, 32)
        self.conv10 = DoubleConv(32, 64)
        self.conv20 = DoubleConv(64, 128)
        self.conv30 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # nested decoder路径（到 level‑3 为止）
        self.conv01 = DoubleConv(32 + 64, 32)
        self.conv11 = DoubleConv(64 + 128, 64)
        self.conv21 = DoubleConv(128 + 256, 128)

        self.conv02 = DoubleConv(32*2 + 64, 32)
        self.conv12 = DoubleConv(64*2 + 128, 64)

        self.conv03 = DoubleConv(32*3 + 64, 32)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # encoder
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        # decoder嵌套
        x01 = self.conv01(torch.cat([x00, self.up(x10)], 1))
        x11 = self.conv11(torch.cat([x10, self.up(x20)], 1))
        x21 = self.conv21(torch.cat([x20, self.up(x30)], 1))
        x02 = self.conv02(torch.cat([x00, x01, self.up(x11)], 1))
        x12 = self.conv12(torch.cat([x10, x11, self.up(x21)], 1))
        x03 = self.conv03(torch.cat([x00, x01, x02, self.up(x12)], 1))
        return self.final(x03)
# ----------------------------------------------------------------------

# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = UNetPlusPlus().to(DEVICE)

    train_loader = DataLoader(
        PairedMRIDataset(ROOT_DIR, "train"),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4      
    )

    # --- 损失：Smooth‑L1 (更鲁棒) ---
    criterion = nn.SmoothL1Loss(beta=0.1)

    best = float("inf")
    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            pred  = model(x)
            loss  = criterion(pred, y)   # ← 使用新损失

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()

        running /= len(train_loader)
        print(f"Ep {ep:02d} | Train {running:.4f}")

        # --- 保存 checkpoint ---
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, f"model_ep{ep}.pt"))
        if running < best:
            best = running
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "best_model.pt"))
            print("✅ Saved best model")

# ---# -------------------------------------------------------------------
# Testing & Metrics
# -------------------------------------------------------------------
def test():
    model = UNetPlusPlus().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pt"), map_location=DEVICE))
    model.eval()

    test_loader = DataLoader(PairedMRIDataset(ROOT_DIR, "test"),
                             batch_size=1, shuffle=False)

    res_dir = os.path.join(SAVE_DIR, "test_pred")
    os.makedirs(res_dir, exist_ok=True)

    results = []
    with torch.no_grad():
        for x3, x7, idx in test_loader:
            x3, x7 = x3.to(DEVICE), x7.to(DEVICE)

            pred = model(x3)

            # ----- slice‑level metrics -----
            l1_val  = nn.L1Loss()(pred, x7).item()
            ssim_val = ssim(pred.squeeze().cpu().numpy(),
                            x7.squeeze().cpu().numpy(), data_range=1.0)
            psnr_val = psnr(pred.squeeze().cpu().numpy(),
                            x7.squeeze().cpu().numpy(), data_range=1.0)
            results.append([idx.item(), l1_val, ssim_val, psnr_val])

            # save visual (input | pred | target)
            vis = torch.cat([x3, pred, x7], dim=-1).squeeze().cpu().numpy()
            plt.imsave(os.path.join(res_dir, f"{idx.item():03d}.png"),
                       vis, cmap="gray")

    # ----- per‑slice CSV -----
    with open(os.path.join(res_dir, "metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["Slice", "L1", "SSIM", "PSNR"]] + results)

    # ----- global averages -----
    arr = np.array(results)
    avg_l1, avg_ssim, avg_psnr = arr[:, 1].mean(), arr[:, 2].mean(), arr[:, 3].mean()

    print(f"\n📊 Average Results for {SAVE_DIR}")
    print(f"  ▸ L1:   {avg_l1:.4f}")
    print(f"  ▸ SSIM: {avg_ssim:.4f}")
    print(f"  ▸ PSNR: {avg_psnr:.2f} dB")

    # ----- append to overall summary -----
    summary_file = os.path.join(os.path.dirname(SAVE_DIR), "metrics_summary.csv")
    header = ["save_dir", "epochs", "lr", "batch_size", "avg_L1", "avg_SSIM", "avg_PSNR"]
    row = [SAVE_DIR, EPOCHS, LR, BATCH_SIZE, avg_l1, avg_ssim, avg_psnr]

    write_header = not os.path.exists(summary_file)
    with open(summary_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--root_dir", default=ROOT_DIR)
    parser.add_argument("--save_dir", default=SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    # override globals
    ROOT_DIR, SAVE_DIR = args.root_dir, args.save_dir
    EPOCHS, BATCH_SIZE, LR = args.epochs, args.batch_size, args.lr

    if args.mode == "train":
        train()
    else:
        test()
