#!/usr/bin/bash

#SBATCH -J splatpose
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o /ceph_data/chanse0727/repos/SplatPosePlus/experiments_20/logs/slurm-%A-%x.out

# 사용법: sbatch --job-name=g1 train.sh 1
GROUP=${1:-"1"}

declare -A CATEGORY_GROUPS
CATEGORY_GROUPS[1]="01Gorilla 02Unicorn 03Mallard 04Turtle 05Whale"
CATEGORY_GROUPS[2]="06Bird 07Owl 08Sabertooth 09Swan 10Sheep"
CATEGORY_GROUPS[3]="11Pig 12Zalika 13Pheonix 14Elephant 15Parrot"
CATEGORY_GROUPS[4]="16Cat 17Scorpion 18Obesobeso 19Bear 20Puppy"

PROJECT_ROOT=/ceph_data/chanse0727/repos/SplatPosePlus
LOGDIR="${PROJECT_ROOT}/experiments_20/logs"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

for CATEGORY in ${CATEGORY_GROUPS[$GROUP]}; do
    echo "========================================="
    echo "GROUP    : ${GROUP}"
    echo "CATEGORY : ${CATEGORY}"
    echo "START    : $(date)"
    echo "========================================="

    python train_render_eval.py -c "${CATEGORY}" \
        2>&1 | tee "${LOGDIR}/${CATEGORY}.log"

    echo "DONE ${CATEGORY}: $(date)"
done