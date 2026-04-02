#!/usr/bin/bash

#SBATCH -J splatpose-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -w aurora-g3
#SBATCH -o /ceph_data/chanse0727/repos/SplatPosePlus/experiments_1/logs/slurm-%A.out

CATEGORY=${1:-"01Gorilla"}

PROJECT_ROOT=/ceph_data/chanse0727/repos/SplatPosePlus
LOGDIR="${PROJECT_ROOT}/experiments_1/logs"

##### ✅ conda 활성화
source ~/.bashrc
conda activate splatposeplus
#####

##### ✅ wandb 설정
export WANDB_API_KEY="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"

export WANDB_PROJECT="first"
export WANDB_ENTITY="3DGaussianS"
#####

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "========================================="
echo "CATEGORY : ${CATEGORY}"
echo "START    : $(date)"
echo "========================================="

srun --mpi=none python train_render_eval1.py -c "${CATEGORY}" \
    2>&1 | tee "${LOGDIR}/${CATEGORY}.log"

echo "DONE: $(date)"