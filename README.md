# SplatPose+ : Gaussian Parameter Space Anomaly Detection

> **Based on** [SplatPose+: Real-time Image-Based Pose-Agnostic 3D Anomaly Detection](https://arxiv.org/abs/2410.12080) (ECCV 2024 Workshop)  
> **Extended by** Chanse, Chaeyoung — Kyung Hee University

---

## Overview

This repository extends the original **SplatPose+** framework to explore **3D Gaussian Splatting (3DGS)-based anomaly detection** using the Gaussian parameter space directly, rather than relying solely on 2D rendering-based comparison.

The core idea:
- Train 3DGS on **normal images** → extract Gaussian parameters as **Normal GT**
- Train 3DGS on **anomalous images** → extract Gaussian parameters as **Anomaly GT**
- Detect anomalies by measuring **statistical deviation** in Gaussian parameter distributions

![Pipeline Overview](./assets/fig.png)

---

## Key Contributions

### 찬세의 Approach — Anomal 3DGS Re-training
실제 이상 이미지로 3DGS를 재학습하여 이상 Gaussian 파라미터를 추출

```
Normal images → 3DGS training → Normal point_cloud.ply  (GT)
Anomal images → 3DGS training → Anomal point_cloud.ply  (GT)
→ Compare parameter distributions → Anomaly Score
```

### Chaeyoung's Approach — Parameter Manipulation
정상 파라미터를 직접 조작하여 이상 GT를 합성

| Defect | Parameter | Manipulation |
|--------|-----------|-------------|
| Burrs | scale | `scale[mask] *= N` |
| Missing | opacity | `opacity[mask] = -10` |
| Stains | f_dc | `f_dc[mask] = target_val` |

---

## Dataset

**MAD-Sim** (Multi-pose Anomaly Detection — Simulation)
- 20 categories of LEGO toys
- 3 defect types: **Burrs**, **Missing**, **Stains**
- Evaluation: Image AUROC / Pixel AUROC / AUPRO

**Focus categories**: Gorilla, Unicorn, Pig, Cat, Bear

---

## Installation

```bash
# (1) Clone with submodules
git clone https://github.com/chansses/SplatPosePlus.git --recursive
cd SplatPosePlus

# (2) Create environment
export CUDA_HOME=/usr/local/cuda
conda env create --file environment.yml
conda activate splatposeplus
pip install -e submodules/Hierarchical-Localization/

# (3) Download MAD-Sim dataset
gdown 1XlW5v_PCXMH49RSKICkskKjd2A5YDMQq
unzip MAD-Sim.zip

# (4) Download pretrained PAD model
cd PAD_utils
gdown https://drive.google.com/uc\?id\=16FOwaqQE0NGY-1EpfoNlU0cGlHjATV0V
unzip model.zip
cd ..
```

---

## Usage

### 1. Normal GT 생성 (정상 3DGS 학습)

```bash
# 단일 카테고리
python train_render_eval.py -c 01Gorilla

# seed 변형으로 다양한 정상 GT 생성
python train_render_eval.py -c 01Gorilla -seed 1 -skip_loc
python train_render_eval.py -c 01Gorilla -seed 2 -skip_loc
python train_render_eval.py -c 01Gorilla -seed 3 -skip_loc

# 전체 카테고리 (4개 GPU 병렬)
sbatch --job-name=g1 experiments_20/train.sh 1
sbatch --job-name=g2 experiments_20/train.sh 2
sbatch --job-name=g3 experiments_20/train.sh 3
sbatch --job-name=g4 experiments_20/train.sh 4
```

### 2. Anomaly GT 생성 (이상 3DGS 학습)

```bash
# 단일 카테고리, 전체 defect
python anomal_parameter_extract/anomal_parameter.py -c 01Gorilla

# 특정 defect만
python anomal_parameter_extract/anomal_parameter.py -c 01Gorilla -defect Burrs

# 5개 카테고리 동시 제출
bash anomal_parameter_extract/submit_all.sh
```

### 3. 파라미터 시각화

```bash
# 정상 파라미터 3D 시각화
python anomal_parameter_extract/visualize_params.py -c 01Gorilla

# 정상 vs 이상 비교
python anomal_parameter_extract/compare_3d.py -c 01Gorilla
```

---

## Experiment Structure

```
SplatPosePlus/
├── train_render_eval.py          # 메인 학습/평가 스크립트
├── anomal_parameter_extract/     # 이상 GT 생성 및 시각화
│   ├── anomal_parameter.py       # 이상 3DGS 학습
│   ├── compare_3d.py             # 정상 vs 이상 3D 비교 시각화
│   ├── visualize_params.py       # 파라미터 분포 시각화
│   └── submit_all.sh             # 5개 카테고리 일괄 제출
├── experiments_20/               # 전체 20개 카테고리 실험
│   ├── train.sh                  # SLURM 학습 스크립트
│   └── logs/                     # 실험 로그
└── results/                      # 학습 결과 (GT 저장)
    ├── 01Gorilla/                 # seed 0
    ├── 01Gorilla_seed1/           # seed 1
    ├── 01Gorilla_seed2/           # seed 2
    ├── 01Gorilla_seed3/           # seed 3
    └── 01Gorilla_test_Burrs/      # 이상 GT
```

---

## Results (01Gorilla, seed 0)

| Metric | Score |
|--------|-------|
| Pixel ROCAUC | 0.997 |
| AUPRO | 0.958 |
| Image ROCAUC | 0.920 |

---

## Credits

- [SplatPose+](https://github.com/Yizhe-Liu/SplatPosePlus) — Original paper & code
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection](https://github.com/EricLee0224/PAD)
- [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)

---

## License

The gaussian-splatting module is licensed under the respective "Gaussian-Splatting License" found in LICENSE.md.