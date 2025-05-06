#!/usr/bin/env python3
import os
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- CLI ----------
def parse_opts():
    p = argparse.ArgumentParser("PatchGAN MRI Translator")
    p.add_argument("--mode", choices=["train", "test"], required=True)
    p.add_argument("--root_dir",   required=True)
    p.add_argument("--save_dir",   required=True)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--noise_std",  type=float, default=0.0)
    p.add_argument("--slice_min",  type=int,   default=160)
    p.add_argument("--slice_max",  type=int,   default=260)
    # λ weights
    p.add_argument("--lambda_l1",     type=float, default=10.0)
    p.add_argument("--lambda_fm",     type=float, default=10.0)
    p.add_argument("--lambda_gan",    type=float, default=1.0)
    p.add_argument("--lambda_cycle",  type=float, default=1.0)
    p.add_argument("--lambda_percep", type=float, default=2.0)
    p.add_argument("--lambda_gp",     type=float, default=10.0)
    # optional global summaries
    p.add_argument("--summary_csv",
                   help="append avg metrics to this global CSV")
    return p.parse_args()

# ---------- Dataset ----------
class PairedMRIDataset(Dataset):
    def __init__(self, root_dir, mode,
                 slice_range=(160,260), noise_std=0.0):
        self.noise = noise_std if mode=="train" else 0.0
        subs = [f"sub-{i:02d}" for i in range(1,10)] if mode=="train" else ["sub-10"]
        self.samples = []
        for sid in subs:
            p3 = os.path.join(root_dir, f"{sid}_ses-1_T1w_defaced_registered.nii")
            p7 = os.path.join(root_dir, f"{sid}_ses-2_T1w_defaced_registered.nii")
            if not (os.path.exists(p3) and os.path.exists(p7)): continue
            v3, v7 = nib.load(p3).get_fdata(), nib.load(p7).get_fdata()
            if v3.shape!=v7.shape: continue
            for i in range(*slice_range):
                self.samples.append((v3[:,:,i], v7[:,:,i], i))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s3,s7,i = self.samples[idx]
        s3,s7 = s3[:256,:256], s7[:256,:256]
        norm = lambda x: ((x - x.min())/(x.max()-x.min()+1e-8)).astype(np.float32)
        s3, s7 = norm(s3), norm(s7)
        if self.noise>0:
            n = np.random.normal(0, self.noise, s3.shape).astype(np.float32)
            s3, s7 = np.clip(s3+n,0,1), np.clip(s7+n,0,1)
        return torch.from_numpy(s3[None]), torch.from_numpy(s7[None]), i

# ---------- Networks ----------
class ResBlock(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,3,1,1), nn.BatchNorm2d(ch))
    def forward(self,x): return F.relu(x + self.net(x))

class Down(nn.Module):
    def __init__(self,ic,oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic,oc,4,2,1), nn.BatchNorm2d(oc), nn.LeakyReLU(0.2,True))
    def forward(self,x): return self.net(x)

class Up(nn.Module):
    def __init__(self,ic,oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True),
            nn.Conv2d(ic,oc,3,1,1), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self,x): return self.net(x)

class ImprovedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1=Down(1,64);   self.e2=Down(64,128)
        self.e3=Down(128,256); self.e4=Down(256,512)
        self.res=ResBlock(512)
        self.u4=Up(1024,256);  self.u3=Up(512,128)
        self.u2=Up(256,64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128,1,4,2,1), nn.Tanh())
    def encode(self,x):
        e1=self.e1(x); e2=self.e2(e1)
        e3=self.e3(e2); e4=self.e4(e3)
        return [e1,e2,e3,self.res(e4)]
    def decode(self,fs,ft):
        d=self.u4(torch.cat([fs[3], ft[3]],1))
        d=self.u3(torch.cat([d, fs[2]],1))
        d=self.u2(torch.cat([d, fs[1]],1))
        return self.final(torch.cat([d, fs[0]],1))
    def forward(self,x3,x7):
        return self.decode(self.encode(x3), self.encode(x7))

class PatchD(nn.Module):
    def __init__(self,in_ch=2):
        super().__init__()
        def blk(ic,oc,s,bn=True):
            m=[nn.Conv2d(ic,oc,4,s,1)]
            if bn: m.append(nn.BatchNorm2d(oc))
            m.append(nn.LeakyReLU(0.2,True)); return m
        self.m = nn.Sequential(
            *blk(in_ch,64,2,False), *blk(64,128,2),
            *blk(128,256,2), nn.Conv2d(256,1,4,1,1))
    def forward(self,x,y): return self.m(torch.cat([x,y],1))

# perceptual
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval().to(DEVICE)
for p in vgg.parameters(): p.requires_grad=False
def perc(f,r):
    return F.mse_loss(vgg(f.expand(-1,3,-1,-1)), vgg(r.expand(-1,3,-1,-1)))

def gp_loss(D,x,y_real,y_fake,λ_gp):
    a = torch.rand(x.size(0),1,1,1,device=x.device)
    inter = (a*y_real + (1-a)*y_fake).requires_grad_(True)
    d = D(x,inter)
    g = torch.autograd.grad(d, inter, torch.ones_like(d), create_graph=True)[0]
    return λ_gp * ((g.view(g.size(0),-1).norm(2,1)-1)**2).mean()

# ---------- train / validate / test ----------
def train(args):
    from pathlib import Path
    import csv
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    # Prepare save directory and validation log
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    val_log = Path(args.save_dir) / "val_log.csv"
    with open(val_log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "avg_SSIM", "avg_PSNR"])

    # DataLoader for training
    loader = DataLoader(
        PairedMRIDataset(
            args.root_dir, "train",
            slice_range=(args.slice_min, args.slice_max),
            noise_std=args.noise_std
        ),
        batch_size=args.batch_size,
        shuffle=True
    )

    # Initialize networks and optimizers
    G3to7 = ImprovedUNet().to(DEVICE)
    G7to3 = ImprovedUNet().to(DEVICE)
    D     = PatchD().to(DEVICE)
    optG = torch.optim.Adam(
        list(G3to7.parameters()) + list(G7to3.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Training loop
    for ep in range(1, args.epochs + 1):
        g_sum = d_sum = 0.0
        for x3, x7, _ in loader:
            x3, x7 = x3.to(DEVICE), x7.to(DEVICE)

            # Discriminator step
            fake7 = G3to7(x3, x7).detach()
            d_real = D(x3, x7)
            d_fake = D(x3, fake7)
            lossD = -d_real.mean() + d_fake.mean() + \
                    gp_loss(D, x3, x7, fake7, args.lambda_gp)
            optD.zero_grad()
            lossD.backward()
            optD.step()

            # Generator step
            fake7 = G3to7(x3, x7)
            rec3  = G7to3(fake7, x3)
            loss_gan = -D(x3, fake7).mean()
            lossG = (
                args.lambda_gan   * loss_gan +
                args.lambda_l1    * F.l1_loss(fake7, x7) +
                args.lambda_percep * perc(fake7, x7) +
                args.lambda_cycle * F.l1_loss(rec3, x3)
            )
            optG.zero_grad()
            lossG.backward()
            optG.step()

            g_sum += lossG.item()
            d_sum += lossD.item()

        print(f"Epoch {ep:02d}/{args.epochs}  G={g_sum/len(loader):.4f}  D={d_sum/len(loader):.4f}")

        # Checkpoint generator for this epoch
        torch.save(
            {"G3to7": G3to7.state_dict()},
            Path(args.save_dir) / "last_joint.pth"
        )

        # Validate immediately after saving
        test(args)
        dfv = pd.read_csv(Path(args.save_dir) / "test_3to7" / "metrics.csv")
        avg_ssim = dfv["SSIM"].mean()
        avg_psnr = dfv["PSNR"].mean()

        # Append to val_log.csv
        with open(val_log, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{avg_ssim:.4f}", f"{avg_psnr:.4f}"])

    # Final checkpoint
    torch.save(
        {"G3to7": G3to7.state_dict()},
        Path(args.save_dir) / "last_joint_final.pth"
    )


def stretch(x, low=2, high=98):
    p1,p2 = np.percentile(x, (low, high))
    return np.clip((x-p1)/(p2-p1+1e-8), 0,1)

def test(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    # load model
    G = ImprovedUNet().to(DEVICE)
    ck = torch.load(Path(args.save_dir)/"last_joint.pth", map_location=DEVICE)
    G.load_state_dict(ck["G3to7"])
    G.eval()

    # dataloader
    loader = DataLoader(
        PairedMRIDataset(args.root_dir, "test",
                         slice_range=(args.slice_min,args.slice_max)),
        batch_size=1, shuffle=False)

    # prepare
    res_dir = Path(args.save_dir)/"test_3to7"
    res_dir.mkdir(exist_ok=True)
    metrics = []

    with torch.no_grad():
        for x3,x7,i in loader:
            x3,x7 = x3.to(DEVICE), x7.to(DEVICE)
            pred = G(x3,x7)
            p_np = pred.add(1).mul(0.5).squeeze().cpu().numpy()
            t_np = x7.add(1).mul(0.5).squeeze().cpu().numpy()
            x3_np= x3.squeeze().cpu().numpy()

            # metrics
            l1 = F.l1_loss(pred.add(1).mul(0.5), x7.add(1).mul(0.5)).item()
            s = ssim(p_np, t_np, data_range=1.0)
            ps = psnr(p_np, t_np, data_range=1.0)
            metrics.append([i.item(), l1, s, ps])

            # visualization
            viz = np.hstack([stretch(x3_np), stretch(p_np), stretch(t_np)])
            plt.imsave(res_dir/f"{i.item():03d}.png",
                       viz, cmap="gray", vmin=0, vmax=1)

    # save slice metrics
    with open(res_dir/"metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows([["Slice","L1","SSIM","PSNR"]] + metrics)

    # compute averages
    arr = np.array(metrics)
    avg_l1  = arr[:,1].mean()
    avg_ssim= arr[:,2].mean()
    avg_psnr= arr[:,3].mean()
    print(f"\n📊 Average L1 {avg_l1:.4f} | SSIM {avg_ssim:.4f} | PSNR {avg_psnr:.2f} dB\n")

    # append to global summary
    if args.summary_csv:
        hdr = not Path(args.summary_csv).exists()
        with open(args.summary_csv, "a", newline="") as f:
            w=csv.writer(f)
            if hdr:
                w.writerow([
                    "save_dir","lambda_l1","lambda_fm","lambda_gan",
                    "lambda_cycle","lambda_percep","lambda_gp",
                    "epochs","lr","batch_size","noise_std",
                    "avg_L1","avg_SSIM","avg_PSNR"
                ])
            w.writerow([
                args.save_dir,
                args.lambda_l1, args.lambda_fm, args.lambda_gan,
                args.lambda_cycle, args.lambda_percep, args.lambda_gp,
                args.epochs, args.lr, args.batch_size, args.noise_std,
                avg_l1, avg_ssim, avg_psnr
            ])

# ---------- main ----------
if __name__=="__main__":
    args = parse_opts()
    if args.mode=="train":
        train(args)
    else:
        test(args)
