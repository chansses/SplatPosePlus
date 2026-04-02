#!/usr/bin/bash

CATEGORIES=("01Gorilla" "02Unicorn" "11Pig" "16Cat" "19Bear")

for CAT in "${CATEGORIES[@]}"; do
    sbatch --job-name="${CAT}-anomal" \
           train_anomal.sh "${CAT}"
    echo "제출 완료: ${CAT}"
done