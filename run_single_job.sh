
set -euo pipefail

# ───── 基础路径 & 公共超参 ─────
ROOT_DIR="/home/yaxin/My_Files/3t-to-7t-mri/Aligned"
SAVE_BASE="/home/yaxin/My_Files/3t-to-7t-mri/6loss_grid_random"
mkdir -p "$SAVE_BASE"

BATCH_SIZE=4
LR=1e-4
NOISE_STD=0.01
EPOCHS=3          # 粗筛：3 epoch

# ───── 超参取值 ─────
L1S=(5.0 10.0)
FMS=(5.0 15.0)
GANS=(0.25 0.5 1.0)
CYCLES=(0.0 0.5 1.0)
PERCEPS=(0.0 0.1 0.2)
GPS=(0.0 2.0 5.0)

# ───── 全局汇总 CSV ─────
SUMMARY="$SAVE_BASE/metrics_summary.csv"
rm -f "$SUMMARY"   # 重新跑时先清空
echo "save_dir,lambda_l1,lambda_fm,lambda_gan,lambda_cycle,lambda_percep,lambda_gp,epochs,lr,batch_size,noise_std,avg_L1,avg_SSIM,avg_PSNR" \
  > "$SUMMARY"

# ───── 1) 生成全部组合 ─────
declare -a COMBOS
for L1 in "${L1S[@]}"; do
  for FM in "${FMS[@]}"; do
    for GAN in "${GANS[@]}"; do
      for CYC in "${CYCLES[@]}"; do
        for PER in "${PERCEPS[@]}"; do
          for GP in "${GPS[@]}"; do
            COMBOS+=("${L1}:${FM}:${GAN}:${CYC}:${PER}:${GP}")
          done
        done
      done
    done
  done
done

# ───── 2) 随机抽 30 组 ─────
SAMPLE_SIZE=50
mapfile -t SELECTED < <(printf '%s\n' "${COMBOS[@]}" | shuf -n "$SAMPLE_SIZE")

echo "🚀  Random sweep:  ${SAMPLE_SIZE}/${#COMBOS[@]} combos,  ${EPOCHS} epoch each"
for COMB in "${SELECTED[@]}"; do
  IFS=':' read -r L1 FM GAN CYC PER GP <<< "$COMB"

  RUN_DIR="$SAVE_BASE/l1_${L1}_fm_${FM}_gan_${GAN}_cyc_${CYC}_per_${PER}_gp_${GP}"
  echo "▶  L1=${L1}  FM=${FM}  GAN=${GAN}  CYCLE=${CYC}  PER=${PER}  GP=${GP}"

  python test.py --mode train \
    --root_dir     "$ROOT_DIR" \
    --save_dir     "$RUN_DIR" \
    --epochs       "$EPOCHS" \
    --lr           "$LR" \
    --batch_size   "$BATCH_SIZE" \
    --noise_std    "$NOISE_STD" \
    --lambda_l1    "$L1" \
    --lambda_fm    "$FM" \
    --lambda_gan   "$GAN" \
    --lambda_cycle "$CYC" \
    --lambda_percep "$PER" \
    --lambda_gp    "$GP" \
    --summary_csv  "$SUMMARY"
done

echo "✅  Sweep finished.  Results collected in:  $SUMMARY"
