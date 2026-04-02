import os
import sys
import torch
import json
import numpy as np
import shutil
import wandb
from pathlib import Path
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image

# ── 경로 설정 (anomal_parameter_extract/ 안에서 실행되므로) ──────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gaussian_splatting.train import *

# ── Args ─────────────────────────────────────────────────────────
pre_parser = ArgumentParser(description="Test Gaussian Parameter Extraction")
pre_parser.add_argument("-c",         type=str, default="01Gorilla", help="category name")
pre_parser.add_argument("-defect",    type=str, default=None,        help="특정 defect 타입. None이면 전체")
pre_parser.add_argument("-iters",     type=int, default=15000,       help="3DGS 학습 iterations")
pre_parser.add_argument("-data_path", type=str, default="MAD-Sim/",  help="dataset root (PROJECT_ROOT 기준 상대경로)")
pre_parser.add_argument("-seed",      type=int, default=0)
lego_args = pre_parser.parse_args()

# 모든 경로를 PROJECT_ROOT 기준으로 절대경로화
data_root   = PROJECT_ROOT / lego_args.data_path
data_path   = data_root / lego_args.c
scene_path  = Path(data_path)
result_root = PROJECT_ROOT / "results"

print(f"PROJECT_ROOT : {PROJECT_ROOT}")
print(f"data_path    : {data_path}")
print(f"result_root  : {result_root}")

# ── Step 1: 카메라 intrinsic 로드 ────────────────────────────────
print("\nStep 1: 카메라 intrinsic 로드")
with open(scene_path / 'transforms.json') as f:
    orig_json = json.load(f)

first_img   = Image.open(scene_path / 'train/good/0.png')
(w, h)      = first_img.size
cam_angle_x = orig_json['camera_angle_x']
focal       = 0.5 * w / np.tan(0.5 * cam_angle_x)
print(f"  이미지 크기: {w}x{h}, focal: {focal:.2f}")

# ── Step 2: query_poses.txt 로드 ─────────────────────────────────
print("\nStep 2: query_poses.txt 로드")
poses = {}
with open(scene_path / 'query_poses.txt') as fp:
    for line in fp.readlines():
        name, qw, qx, qy, qz, x, y, z = line.split()
        qw, qx, qy, qz = map(float, [qw, qx, qy, qz])
        x,  y,  z      = map(float, [x,  y,  z ])

        R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t_w2c = np.array([x, y, z])
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3,  3] = t_c2w
        c2w[:3, 1:3] *= -1  # COLMAP → OpenGL/Blender

        poses[name] = c2w

print(f"  로드된 포즈 수: {len(poses)}")

# ── Step 3: defect 타입 결정 ─────────────────────────────────────
test_dir     = scene_path / 'test'
defect_types = sorted([d.name for d in test_dir.iterdir()
                        if d.is_dir() and d.name != 'good'])  # good 제외
if lego_args.defect:
    defect_types = [lego_args.defect]

print(f"\n처리할 defect 타입: {defect_types}")

# ── Step 4: defect별 3DGS 학습 ───────────────────────────────────
for defect in defect_types:
    print(f"\n{'='*55}")
    print(f"  CATEGORY : {lego_args.c}")
    print(f"  DEFECT   : {defect}")
    print(f"{'='*55}")

    # 포즈 매칭된 이미지 수집
    frames = []
    for img_file in sorted((test_dir / defect).glob('*.png')):
        key = f"test/{defect}/{img_file.name}"
        if key in poses:
            frames.append({
                "file_path":        f"test/{defect}/{img_file.stem}",
                "transform_matrix": poses[key].tolist()
            })

    print(f"  포즈 매칭된 이미지 수: {len(frames)}")
    if len(frames) < 5:
        print(f"  이미지 수 부족 ({len(frames)}개) → 스킵")
        continue

    # 임시 데이터 디렉토리: MAD-Sim/01Gorilla_test_Burrs/
    temp_dir = data_root / f"{lego_args.c}_test_{defect}"
    temp_dir.mkdir(exist_ok=True)

    transforms = {"camera_angle_x": cam_angle_x, "frames": frames}
    with open(temp_dir / 'transforms_train.json', 'w') as f:
        json.dump(transforms, f, indent=2)
    with open(temp_dir / 'transforms_test.json', 'w') as f:
        json.dump(transforms, f, indent=2)

    # 이미지 심링크: temp_dir/test → MAD-Sim/01Gorilla/test
    link_test = temp_dir / 'test'
    if not link_test.exists():
        os.symlink((scene_path / 'test').resolve(), link_test)
        print(f"  심링크: {link_test} → {(scene_path / 'test').resolve()}")

    # 초기 포인트 클라우드 복사
    src_ply = scene_path / 'points3d.ply'
    dst_ply = temp_dir   / 'points3d.ply'
    if not dst_ply.exists():
        shutil.copy(src_ply, dst_ply)
        print(f"  points3d.ply 복사 완료")

    # 결과 저장 경로: results/01Gorilla_test_Burrs/
    result_dir = str(result_root / f"{lego_args.c}_test_{defect}")
    os.makedirs(result_dir, exist_ok=True)
    print(f"  결과 저장 경로: {result_dir}")

    # ── wandb 초기화 (defect별로 init/finish) ───────────────────
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "anomal_extract"),
        entity=os.environ.get("WANDB_ENTITY", "3DGaussianS"),
        name=f"{lego_args.c}_test_{defect}",
        config={
            "category": lego_args.c,
            "defect":   defect,
            "iters":    lego_args.iters,
            "seed":     lego_args.seed,
        }
    )

    # ── 3DGS 학습 ────────────────────────────────────────────────
    training_args = [
        "-w",
        "-s", str(temp_dir),
        "-m", result_dir,
        "--iterations",             str(lego_args.iters),
        "--densification_interval", "1000"
    ]

    gs_parser = ArgumentParser(description="3DGS Training")
    lp = ModelParams(gs_parser)
    op = OptimizationParams(gs_parser)
    pp = PipelineParams(gs_parser)
    gs_parser.add_argument('--ip',                    type=str,  default="127.0.0.1")
    gs_parser.add_argument('--port',                  type=int,  default=6009)
    gs_parser.add_argument('--debug_from',            type=int,  default=-1)
    gs_parser.add_argument('--detect_anomaly',        action='store_true', default=False)
    gs_parser.add_argument("--test_iterations",       nargs="+", type=int, default=[lego_args.iters])
    gs_parser.add_argument("--save_iterations",       nargs="+", type=int, default=[lego_args.iters])
    gs_parser.add_argument("--quiet",                 action="store_true")
    gs_parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    gs_parser.add_argument("--start_checkpoint",      type=str,  default=None)
    train_args = gs_parser.parse_args(training_args)
    train_args.save_iterations.append(train_args.iterations)

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()

    safe_state(True, lego_args.seed)
    torch.autograd.set_detect_anomaly(False)
    training(
        lp.extract(train_args), op.extract(train_args), pp.extract(train_args),
        train_args.test_iterations, train_args.save_iterations,
        train_args.checkpoint_iterations, train_args.start_checkpoint,
        train_args.debug_from, 1.0
    )

    end.record()
    torch.cuda.synchronize()
    train_time = start.elapsed_time(end) / 1000

    ply_path = f"{result_dir}/point_cloud/iteration_{lego_args.iters}/point_cloud.ply"
    print(f"  학습 시간: {train_time:.1f}초")
    print(f"  PLY 저장 : {ply_path}")

    wandb.log({"time/train_sec": train_time})
    wandb.finish()  # ← defect 루프마다 종료

print("\n" + "="*55)
print("전체 완료!")
print("="*55)
print(f"\n결과 확인:")
print(f"  정상 GT : results/{lego_args.c}/point_cloud/iteration_{lego_args.iters}/point_cloud.ply")
for defect in defect_types:
    print(f"  이상({defect}) : results/{lego_args.c}_test_{defect}/point_cloud/iteration_{lego_args.iters}/point_cloud.ply")