#!/usr/bin/bash

#SBATCH -J splatpose-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -w aurora-g3
#SBATCH -o /ceph_data/chanse0727/repos/SplatPosePlus/experiments_1/logs/slurm-%A.out  # ← 절대경로


# Usage: sbatch train.sh [CATEGORY]
# 이건, 모든 카테고리를 동시에 학습하는게 아니라 단일 카테고리만 학습!
CATEGORY=${1:-"01Gorilla"}   # 기본값은 "01Gorilla"로 설정, 필요에 따라 변경 가능

PROJECT_ROOT=/ceph_data/chanse0727/repos/SplatPosePlus
LOGDIR="${PROJECT_ROOT}/experiments_1/logs"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "========================================="
echo "CATEGORY : ${CATEGORY}"
echo "START    : $(date)"
echo "========================================="

srun --mpi=none python train_render_eval.py -c "${CATEGORY}" \
    2>&1 | tee "${LOGDIR}/${CATEGORY}.log"

echo "DONE: $(date)"

