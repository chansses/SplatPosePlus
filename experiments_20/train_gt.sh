#!/usr/bin/bash

#SBATCH -J splatpose
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o /ceph_data/chanse0727/repos/SplatPosePlus/experiments_20/logs/slurm-%A-%x.out

GROUP=${1:-"1"}

declare -A CATEGORY_GROUPS
CATEGORY_GROUPS[1]="01Gorilla 02Unicorn 03Mallard 04Turtle 05Whale"
CATEGORY_GROUPS[2]="06Bird 07Owl 08Sabertooth 09Swan 10Sheep"
CATEGORY_GROUPS[3]="11Pig 12Zalika 13Pheonix 14Elephant 15Parrot"
CATEGORY_GROUPS[4]="16Cat 17Scorpion 18Obesobeso 19Bear 20Puppy"

SEEDS=(0 1 2 3)   # seed 0은 이미 완료

PROJECT_ROOT=/ceph_data/chanse0727/repos/SplatPosePlus
LOGDIR="${PROJECT_ROOT}/experiments_20/logs"

source ~/.bashrc
conda activate splatposeplus

export WANDB_API_KEY="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"
export WANDB_PROJECT="normal_gt"
export WANDB_ENTITY="3DGaussianS"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

for CATEGORY in ${CATEGORY_GROUPS[$GROUP]}; do
    for SEED in "${SEEDS[@]}"; do
        echo "========================================="
        echo "CATEGORY : ${CATEGORY}"
        echo "SEED     : ${SEED}"
        echo "START    : $(date)"
        echo "========================================="

        python train_render_eval1.py \
            -c "${CATEGORY}" \
            -seed "${SEED}" \
            -skip_loc \
            2>&1 | tee "${LOGDIR}/${CATEGORY}_seed${SEED}.log"

        echo "DONE ${CATEGORY} seed=${SEED}: $(date)"
    done
done