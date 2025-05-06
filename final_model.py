#!/usr/bin/env python
# === MRI Translation with PatchGAN ===
import os, argparse, csv
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np, matplotlib.pyplot as plt
from torchvision import models
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

ROOT_DIR   = "/home/yaxin/My_Files/3t-to-7t-mri/Aligned"
SAVE_DIR   = "/home/yaxin/My_Files/3t-to-7t-mri/final_model_100"
SLICE_RANGE = (160, 260)
EPOCHS      = 100
LR          = 1e-4
BATCH_SIZE  = 4
NOISE_STD   = 0.005
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# λ
LAMBDA_L1 = 10.0; LAMBDA_PERCEP = 0.0; LAMBDA_CYCLE = 1.0
LAMBDA_FM = 5.0; LAMBDA_GAN    = 0.25; LAMBDA_GP    = 2.0
class PairedMRIDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        super().__init__()
        self.mode  = mode                           # <‑‑ 新增，供后续引用
        self.noise = NOISE_STD if mode == "train" else 0.0

        subs = [f"sub-{i:02d}" for i in range(1, 10)] if mode == "train" else ["sub-10"]
        self.samples = []
        for sid in subs:
            p3 = os.path.join(root_dir, f"{sid}_ses-1_T1w_defaced_registered.nii")
            p7 = os.path.join(root_dir, f"{sid}_ses-2_T1w_defaced_registered.nii")
            if not (os.path.exists(p3) and os.path.exists(p7)):
                continue
            v3, v7 = nib.load(p3).get_fdata(), nib.load(p7).get_fdata()
            if v3.shape != v7.shape:
                continue
            for i in range(*SLICE_RANGE):
                self.samples.append((v3[:, :, i], v7[:, :, i], i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s3, s7, i = self.samples[idx]

        s3 = s3[:256, :256]
        s7 = s7[:256, :256]

        def norm(x):
            return ((x - x.min()) / (x.max() - x.min() + 1e-8)).astype(np.float32)
        s3, s7 = norm(s3), norm(s7)

        if self.noise > 0:
            n = np.random.normal(0, self.noise, s3.shape).astype(np.float32)
            s3 = np.clip(s3 + n, 0, 1)
            s7 = np.clip(s7 + n, 0, 1)

        return (
            torch.from_numpy(s3[None]).float(),
            torch.from_numpy(s7[None]).float(),
            i,
        )

class ResBlock(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1),nn.BatchNorm2d(ch),nn.ReLU(True),
            nn.Conv2d(ch,ch,3,1,1),nn.BatchNorm2d(ch))
    def forward(self,x): return F.relu(x+self.net(x))

class Down(nn.Module):
    def __init__(self,ic,oc):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ic,oc,4,2,1),
                               nn.BatchNorm2d(oc),nn.LeakyReLU(0.2,True))
    def forward(self,x): return self.net(x)

class Up(nn.Module):
    def __init__(self,ic,oc):
        super().__init__()
        self.net=nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True),
                               nn.Conv2d(ic,oc,3,1,1),nn.BatchNorm2d(oc),nn.ReLU(True))
    def forward(self,x): return self.net(x)

class ImprovedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1=Down(1,64); self.e2=Down(64,128); self.e3=Down(128,256); self.e4=Down(256,512)
        self.res=ResBlock(512)
        self.u4=Up(1024,256); self.u3=Up(512,128); self.u2=Up(256,64)
        self.final=nn.Sequential(nn.ConvTranspose2d(128,1,4,2,1),nn.Tanh())
    def encode(self,x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        return [e1,e2,e3,self.res(e4)]
    def decode(self,e3t,e7t):
        d=self.u4(torch.cat([e3t[3],e7t[3]],1))
        d=self.u3(torch.cat([d,e3t[2]],1))
        d=self.u2(torch.cat([d,e3t[1]],1))
        return self.final(torch.cat([d,e3t[0]],1))
    def forward(self,x3,x7):
        return self.decode(self.encode(x3),self.encode(x7))

# PatchGAN 判别器
class PatchDiscriminator(nn.Module):
    def __init__(self,in_ch=2):
        super().__init__()
        def blk(ic,oc,s,bn=True):
            ls=[nn.Conv2d(ic,oc,4,s,1)]
            if bn: ls.append(nn.BatchNorm2d(oc))
            ls.append(nn.LeakyReLU(0.2,True)); return ls
        self.model=nn.Sequential(
            *blk(in_ch,64,2,False),
            *blk(64,128,2),
            *blk(128,256,2),
            nn.Conv2d(256,1,4,1,1))
    def forward(self,x,y): return self.model(torch.cat([x,y],1))

# perceptual loss (VGG16 conv1_2 ~ conv3_3)
vgg=models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval().to(DEVICE)
for p in vgg.parameters(): p.requires_grad=False
percep=lambda f,r:F.mse_loss(vgg(f.expand(-1,3,-1,-1)),vgg(r.expand(-1,3,-1,-1)))

# ---------- 训练 ----------
def gradient_penalty(D,x,y_real,y_fake):
    a=torch.rand(x.size(0),1,1,1,device=x.device)
    inter=(a*y_real+(1-a)*y_fake).requires_grad_(True)
    d=D(x,inter)
    g=torch.autograd.grad(d,inter,torch.ones_like(d),True,True)[0]
    return ((g.view(g.size(0),-1).norm(2,1)-1)**2).mean()

def train():
    os.makedirs(SAVE_DIR,exist_ok=True)
    loader=DataLoader(PairedMRIDataset(ROOT_DIR,"train"),BATCH_SIZE,shuffle=True)
    G3to7,G7to3=ImprovedUNet().to(DEVICE),ImprovedUNet().to(DEVICE)
    D=PatchDiscriminator().to(DEVICE)
    optG=torch.optim.Adam(list(G3to7.parameters())+list(G7to3.parameters()),lr=LR)
    optD=torch.optim.Adam(D.parameters(),lr=LR)
    for ep in range(1,EPOCHS+1):
        G3to7.train();G7to3.train();D.train()
        g_tot=d_tot=0
        for x3,x7,_ in loader:
            x3,x7=x3.to(DEVICE),x7.to(DEVICE)
            f7=G3to7(x3,x7); rec3=G7to3(f7,x3)

            # ----- train D -----
            d_real=D(x3,x7); d_fake=D(x3,f7.detach())
            gp=gradient_penalty(D,x3,x7,f7.detach())
            lossD=-d_real.mean()+d_fake.mean()+LAMBDA_GP*gp
            optD.zero_grad(); lossD.backward(); optD.step()

            # ----- train G -----
            loss_gan=-D(x3,f7).mean()
            lossG=(LAMBDA_GAN*loss_gan +
                   LAMBDA_L1*F.l1_loss(f7,x7) +
                   LAMBDA_PERCEP*percep(f7,x7) +
                   LAMBDA_CYCLE*F.l1_loss(rec3,x3))
            optG.zero_grad(); lossG.backward(); optG.step()

            g_tot+=lossG.item(); d_tot+=lossD.item()
        print(f"Ep {ep:02d} | G {g_tot/len(loader):.4f} | D {d_tot/len(loader):.4f}")
    torch.save({'G3to7':G3to7.state_dict()},os.path.join(SAVE_DIR,"last_joint.pth"))

def stretch(x, low=2, high=98):
    """把灰度分布按给定分位数拉伸到 0‑1，仅用于可视化。"""
    p1, p2 = np.percentile(x, (low, high))
    return np.clip((x - p1) / (p2 - p1 + 1e-8), 0, 1)
def test():
    os.makedirs(SAVE_DIR, exist_ok=True)

    loader = DataLoader(
        PairedMRIDataset(ROOT_DIR, "test"),
        batch_size=1, shuffle=False
    )

    G = ImprovedUNet().to(DEVICE)
    G.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "last_joint.pth"),
                   map_location=DEVICE)["G3to7"]
    )
    G.eval()

    res_dir = os.path.join(SAVE_DIR, "test_3to7")
    os.makedirs(res_dir, exist_ok=True)

    metrics = []
    with torch.no_grad():
        for x3, x7, i in loader:
            x3, x7 = x3.to(DEVICE), x7.to(DEVICE)

            # ---------- 推断 ----------
            pred = G(x3, x7)

            # ---------- 原生 0‑1 张量 ----------
            p_np = ((pred + 1) * 0.5).squeeze().cpu().numpy()
            t_np = ((x7   + 1) * 0.5).squeeze().cpu().numpy()
            x3_np = x3.squeeze().cpu().numpy()   # 已经 0‑1

            # ---------- 指标 ----------
            l1  = F.l1_loss((pred + 1) * 0.5, (x7 + 1) * 0.5).item()
            ssim_v = ssim(p_np, t_np, data_range=1.0)
            psnr_v = psnr(p_np, t_np, data_range=1.0)
            metrics.append([i.item(), l1, ssim_v, psnr_v])

            # ---------- 三连图（可视化拉伸） ----------
            viz = np.hstack([
                stretch(x3_np),      # 输入
                stretch(p_np),       # 预测 (拉伸后)
                stretch(t_np)        # 目标 (拉伸后)
            ])
            plt.imsave(
                os.path.join(res_dir, f"{i.item():03d}.png"),
                viz,
                cmap="gray",
                vmin=0, vmax=1
            )

    # ---------- CSV ----------
    with open(os.path.join(res_dir, "metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["Slice", "L1", "SSIM", "PSNR"]] + metrics)

    # ---------- 全局平均 ----------
    arr = np.array(metrics)
    avg_l1, avg_ssim, avg_psnr = arr[:, 1].mean(), arr[:, 2].mean(), arr[:, 3].mean()
    print(f"\n📊 Average  L1 {avg_l1:.4f} | SSIM {avg_ssim:.4f} | PSNR {avg_psnr:.2f} dB")
# ---------- CLI ----------
if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=["train","test"],required=True)
    args=parser.parse_args()
    if args.mode=="train": train(); test()
    else: test()
