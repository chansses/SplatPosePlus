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

# ── PLY 로드 ──────────────────────────────────────────────────────
def load_ply(ply_path):
    v       = PlyData.read(str(ply_path))['vertex']
    xyz     = np.stack([v['x'], v['y'], v['z']], axis=1)
    opacity = 1 / (1 + np.exp(-v['opacity']))
    scale   = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1))
    C0      = 0.28209479177387814
    rgb     = np.clip(0.5 + C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1), 0, 1)

    # opacity 필터 (투명한 점 제거)
    mask = opacity > 0.1
    return xyz[mask], opacity[mask], scale[mask], rgb[mask]

# ── 대상 수집 ─────────────────────────────────────────────────────
result_root = PROJECT_ROOT / "results"
targets = {}

normal_ply = result_root / args.c / f"point_cloud/iteration_{args.iters}/point_cloud.ply"
if normal_ply.exists():
    targets["Normal"] = normal_ply

for d in sorted(result_root.iterdir()):
    if d.name.startswith(f"{args.c}_test_"):
        name     = d.name.replace(f"{args.c}_test_", "")
        ply_path = d / f"point_cloud/iteration_{args.iters}/point_cloud.ply"
        if ply_path.exists():
            targets[name] = ply_path

print(f"대상: {list(targets.keys())}")
n = len(targets)

# ── 3방향 시점 정의 ───────────────────────────────────────────────
# (elev, azim) 조합으로 3개 시점
view_angles = [
    (20,  -60, "Front"),
    (80,  -90, "Top"),
    (20,   30, "Side"),
]

# ── 색상 모드별 시각화 ────────────────────────────────────────────
for color_mode in ["rgb", "opacity", "scale"]:

    fig = plt.figure(figsize=(5 * n, 4 * len(view_angles)))
    fig.patch.set_facecolor('#111111')
    fig.suptitle(
        f"{args.c}  ·  color={color_mode}",
        color='white', fontsize=14, fontweight='bold', y=1.01
    )

    for col, (label, ply_path) in enumerate(targets.items()):
        xyz, opacity, scale, rgb = load_ply(ply_path)
        N = len(xyz)

        # 색상값 결정
        if color_mode == "rgb":
            c    = rgb
            vmin = vmax = None
        elif color_mode == "opacity":
            c    = opacity
            vmin, vmax = 0, 1
        else:
            c    = scale.mean(axis=1)
            vmin, vmax = c.min(), np.percentile(c, 95)

        for row, (elev, azim, view_name) in enumerate(view_angles):
            ax = fig.add_subplot(
                len(view_angles), n,
                row * n + col + 1,
                projection='3d'
            )
            ax.set_facecolor('#111111')

            if color_mode == "rgb":
                ax.scatter(
                    xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c=c, s=0.4, alpha=0.7, depthshade=True
                )
            else:
                sc = ax.scatter(
                    xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c=c, cmap='plasma' if color_mode == 'opacity' else 'viridis',
                    vmin=vmin, vmax=vmax,
                    s=0.4, alpha=0.7, depthshade=True
                )
                if row == 0:
                    plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1,
                                 label=color_mode)

            ax.view_init(elev=elev, azim=azim)

            # 축 꾸미기
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#333333')
            ax.yaxis.pane.set_edgecolor('#333333')
            ax.zaxis.pane.set_edgecolor('#333333')

            # 제목 (첫 행에만)
            if row == 0:
                color = 'limegreen' if label == 'Normal' else 'tomato'
                mark  = '✓' if label == 'Normal' else '✗'
                ax.set_title(
                    f"{mark} {label}\nN={N:,}",
                    color=color, fontsize=10, fontweight='bold', pad=4
                )

            # 시점 이름 (첫 열에만)
            if col == 0:
                ax.text2D(-0.12, 0.5, view_name,
                          transform=ax.transAxes,
                          color='gray', fontsize=9,
                          va='center', rotation=90)

    plt.tight_layout()
    out = save_dir / f"{args.c}_3d_{color_mode}.png"
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"저장: {out}")

print(f"\n완료 → {save_dir}")