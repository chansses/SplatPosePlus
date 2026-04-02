import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plyfile import PlyData
from argparse import ArgumentParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

parser = ArgumentParser()
parser.add_argument("-c",     type=str, default="01Gorilla")
parser.add_argument("-iters", type=int, default=15000)
args = parser.parse_args()

save_dir = PROJECT_ROOT / "anomal_parameter_extract" / "vis_output"
save_dir.mkdir(exist_ok=True)

# ── PLY 로드 함수 ─────────────────────────────────────────────────
def load_ply(ply_path):
    ply     = PlyData.read(str(ply_path))
    v       = ply['vertex']
    xyz     = np.stack([v['x'], v['y'], v['z']], axis=1)
    opacity = 1 / (1 + np.exp(-v['opacity']))
    scale   = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1))
    C0      = 0.28209479177387814
    colors  = np.clip(0.5 + C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1), 0, 1)
    return xyz, opacity, scale, colors

# ── 대상 목록 수집 ────────────────────────────────────────────────
result_root = PROJECT_ROOT / "results"
targets = {}

# 정상
normal_ply = result_root / args.c / f"point_cloud/iteration_{args.iters}/point_cloud.ply"
if normal_ply.exists():
    targets["Normal"] = normal_ply

# 이상 (01Gorilla_test_Burrs 등)
for d in sorted(result_root.iterdir()):
    if d.name.startswith(f"{args.c}_test_"):
        defect_name = d.name.replace(f"{args.c}_test_", "")
        ply_path    = d / f"point_cloud/iteration_{args.iters}/point_cloud.ply"
        if ply_path.exists():
            targets[defect_name] = ply_path

print(f"시각화 대상: {list(targets.keys())}")

n_targets = len(targets)

# ── Figure: 행=시점(Front/Top/Side), 열=Normal+defects ───────────
views = [
    ("Front (X-Z)", 0, 2),  # (이름, x축 인덱스, y축 인덱스)
    ("Top   (X-Y)", 0, 1),
    ("Side  (Y-Z)", 1, 2),
]

for color_mode in ["rgb", "opacity", "scale"]:
    fig, axes = plt.subplots(
        len(views), n_targets,
        figsize=(5 * n_targets, 4 * len(views))
    )
    if n_targets == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"{args.c}  —  color: {color_mode}",
        fontsize=15, fontweight='bold'
    )

    for col, (label, ply_path) in enumerate(targets.items()):
        xyz, opacity, scale, colors = load_ply(ply_path)
        N = len(xyz)

        # 색상 결정
        if color_mode == "rgb":
            c = colors
            cmap = None
        elif color_mode == "opacity":
            c    = opacity
            cmap = "plasma"
        else:  # scale
            c    = scale.mean(axis=1)
            cmap = "viridis"

        for row, (view_name, xi, yi) in enumerate(views):
            ax = axes[row][col]
            ax.set_facecolor('black')

            if color_mode == "rgb":
                ax.scatter(xyz[:, xi], xyz[:, yi],
                           c=c, s=0.4, alpha=0.6)
            else:
                sc = ax.scatter(xyz[:, xi], xyz[:, yi],
                                c=c, cmap=cmap, s=0.4, alpha=0.6)
                if row == 0:
                    plt.colorbar(sc, ax=ax, shrink=0.8)

            # 제목: 첫 행에만 카테고리명
            if row == 0:
                tag   = "✅ Normal" if label == "Normal" else f"❌ {label}"
                color = "limegreen" if label == "Normal" else "tomato"
                ax.set_title(f"{tag}\nN={N}", color=color, fontsize=11, fontweight='bold')

            # 왼쪽 첫 열에만 시점 표시
            if col == 0:
                ax.set_ylabel(view_name, color='white', fontsize=9)

            ax.tick_params(colors='gray', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')

    plt.tight_layout()
    out_path = save_dir / f"{args.c}_compare_{color_mode}.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()
    print(f"저장: {out_path}")

print(f"\n완료! → {save_dir}")