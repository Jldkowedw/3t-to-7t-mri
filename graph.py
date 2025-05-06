import pandas as pd
from pathlib import Path

base = Path("/home/yaxin/My_Files/3t-to-7t-mri/6loss_grid_random")
df = pd.read_csv(base/"metrics_summary.csv")

top5 = (df.sort_values(["avg_SSIM","avg_PSNR"], ascending=False)
          .head(3)
          .reset_index(drop=True))

top5.to_csv(base/"top3_configs.csv", index=False)
print(top5)

