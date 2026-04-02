#!/usr/bin/bash

#SBATCH -J anomal-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 2-00:00:00
#SBATCH -o /ceph_data/chanse0727/repos/SplatPosePlus/anomal_parameter_extract/logs/slurm-%A-%x.out

CATEGORY=${1:-"01Gorilla"}

PROJECT_ROOT=/ceph_data/chanse0727/repos/SplatPosePlus

source ~/.bashrc
conda activate splatposeplus

export WANDB_API_KEY="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"
export WANDB_PROJECT="anomal_extract"
export WANDB_ENTITY="3DGaussianS"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

cd "${PROJECT_ROOT}"

echo "========================================="
echo "CATEGORY : ${CATEGORY}"
echo "START    : $(date)"
echo "========================================="

python anomal_parameter_extract/anomal_parameter.py -c "${CATEGORY}" \
    2>&1 | tee "anomal_parameter_extract/logs/${CATEGORY}_anomal.log"

echo "DONE: $(date)"