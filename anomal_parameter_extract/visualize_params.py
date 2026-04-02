import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from plyfile import PlyData
from argparse import ArgumentParser

# ── 경로 설정 ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Args ─────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("-c",      type=str, default="01Gorilla", help="category name")
parser.add_argument("-defect", type=str, default=None,        help="defect 타입. None이면 정상")
parser.add_argument("-iters",  type=int, default=15000)
args = parser.parse_args()

# ── PLY 경로 결정 ─────────────────────────────────────────────────
if args.defect:
    ply_path = PROJECT_ROOT / f"results/{args.c}_test_{args.defect}/point_cloud/iteration_{args.iters}/point_cloud.ply"
    title_prefix = f"{args.c} - {args.defect} (이상)"
    save_name    = f"{args.c}_test_{args.defect}"
else:
    ply_path = PROJECT_ROOT / f"results/{args.c}/point_cloud/iteration_{args.iters}/point_cloud.ply"
    title_prefix = f"{args.c} - Normal (정상)"
    save_name    = f"{args.c}_normal"

save_dir = PROJECT_ROOT / "anomal_parameter_extract" / "vis_output"
save_dir.mkdir(exist_ok=True)

print(f"PLY 경로: {ply_path}")

# ── PLY 로드 ─────────────────────────────────────────────────────
ply  = PlyData.read(str(ply_path))
v    = ply['vertex']
N    = len(v)
print(f"총 Gaussian 수: {N}")

# ── 파라미터 추출 ─────────────────────────────────────────────────
xyz      = np.stack([v['x'], v['y'], v['z']], axis=1)          # (N, 3)
opacity  = 1 / (1 + np.exp(-v['opacity']))                     # sigmoid
scale    = np.exp(np.stack([v['scale_0'],
                             v['scale_1'],
                             v['scale_2']], axis=1))            # exp (log→linear)
rot      = np.stack([v['rot_0'], v['rot_1'],
                     v['rot_2'], v['rot_3']], axis=1)           # quaternion
f_dc     = np.stack([v['f_dc_0'], v['f_dc_1'],
                     v['f_dc_2']], axis=1)                      # SH DC
f_rest   = np.stack([v[f'f_rest_{i}'] for i in range(45)],
                     axis=1)                                    # SH high-order

# ── 색상 복원 (DC만) ──────────────────────────────────────────────
C0     = 0.28209479177387814
colors = np.clip(0.5 + C0 * f_dc, 0, 1)                       # (N, 3) RGB

# ── Figure 1: XYZ ────────────────────────────────────────────────
fig1 = plt.figure(figsize=(20, 5))
fig1.suptitle(f"[1] XYZ - 3D Position  (N={N})", fontsize=13, fontweight='bold')

ax3d = fig1.add_subplot(141, projection='3d')
sc   = ax3d.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
                    c=opacity, cmap='plasma', s=0.3, alpha=0.5)
plt.colorbar(sc, ax=ax3d, shrink=0.5, label='opacity')
ax3d.set_title("3D Point Cloud\n(color=opacity)")
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')

for idx, (axis_data, label, color) in enumerate(zip(
        [xyz[:,0], xyz[:,1], xyz[:,2]], ['X','Y','Z'],
        ['red','green','blue'])):
    ax = fig1.add_subplot(1, 4, idx+2)
    ax.hist(axis_data, bins=80, color=color, alpha=0.7)
    ax.axvline(axis_data.mean(), color='k', linestyle='--', label=f'mean={axis_data.mean():.3f}')
    ax.set_title(f"{label}-axis Distribution\nmin={axis_data.min():.3f} max={axis_data.max():.3f}")
    ax.set_xlabel(label); ax.set_ylabel('count')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(save_dir / f"{save_name}_1_xyz.png", dpi=120, bbox_inches='tight')
plt.close()
print("저장: vis_1_xyz.png")

# ── Figure 2: Scale ───────────────────────────────────────────────
fig2, axes = plt.subplots(1, 4, figsize=(20, 4))
fig2.suptitle("[2] Scaling - Gaussian Size", fontsize=13, fontweight='bold')

scale_labels  = ['Scale X', 'Scale Y', 'Scale Z', 'Scale Mean (3-axis)']
scale_data    = [scale[:,0], scale[:,1], scale[:,2], scale.mean(axis=1)]
scale_colors  = ['red', 'green', 'blue', 'purple']

for ax, data, label, color in zip(axes, scale_data, scale_labels, scale_colors):
    ax.hist(data, bins=80, color=color, alpha=0.7)
    ax.axvline(data.mean(), color='k', linestyle='--', label=f'mean={data.mean():.4f}')
    ax.set_title(f"{label}\nmin={data.min():.4f} max={data.max():.4f}")
    ax.set_xlabel('scale'); ax.set_ylabel('count')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(save_dir / f"{save_name}_2_scaling.png", dpi=120, bbox_inches='tight')
plt.close()
print("저장: vis_2_scaling.png")

# ── Figure 3: Opacity ─────────────────────────────────────────────
fig3, axes = plt.subplots(1, 3, figsize=(18, 4))
fig3.suptitle("[3] Opacity", fontsize=13, fontweight='bold')

# 분포
axes[0].hist(opacity, bins=80, color='steelblue', alpha=0.7)
axes[0].axvline(opacity.mean(), color='k', linestyle='--',
                label=f'mean={opacity.mean():.4f}')
axes[0].set_title(f"Opacity Distribution\nmin={opacity.min():.4f} max={opacity.max():.4f}")
axes[0].set_xlabel('opacity'); axes[0].set_ylabel('count')
axes[0].legend(fontsize=8)

# CDF
sorted_op = np.sort(opacity)
axes[1].plot(sorted_op, np.linspace(0, 1, N), color='steelblue')
axes[1].set_title("Opacity Cumulative Distribution (CDF)")
axes[1].set_xlabel('opacity'); axes[1].set_ylabel('cumulative ratio')
axes[1].grid(alpha=0.3)

# Active Gaussians per threshold
thresholds   = np.linspace(0, 1, 100)
active_count = [(opacity > t).sum() for t in thresholds]
axes[2].plot(thresholds, active_count, color='steelblue')
axes[2].set_title("Active Gaussians per Threshold")
axes[2].set_xlabel('opacity threshold'); axes[2].set_ylabel('count')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / f"{save_name}_3_opacity.png", dpi=120, bbox_inches='tight')
plt.close()
print("저장: vis_3_opacity.png")

# ── Figure 4: Rotation (Quaternion) ──────────────────────────────
fig4, axes = plt.subplots(1, 4, figsize=(20, 4))
fig4.suptitle("[4] Rotation - Quaternion (qw, qx, qy, qz)", fontsize=13, fontweight='bold')

rot_labels = ['qw', 'qx', 'qy', 'qz']
rot_colors = ['gold', 'red', 'green', 'blue']

for ax, data, label, color in zip(axes, rot.T, rot_labels, rot_colors):
    ax.hist(data, bins=80, color=color, alpha=0.7)
    ax.axvline(data.mean(), color='k', linestyle='--',
               label=f'mean={data.mean():.4f}')
    ax.set_title(f"{label}\nstd={data.std():.4f}")
    ax.set_xlabel(label); ax.set_ylabel('count')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(save_dir / f"{save_name}_4_rotation.png", dpi=120, bbox_inches='tight')
plt.close()
print("저장: vis_4_rotation.png")

# ── Figure 5: Spherical Harmonics ────────────────────────────────
fig5 = plt.figure(figsize=(20, 8))
fig5.suptitle("[5] Spherical Harmonics (features) - Color Representation",
              fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig5)

# DC R/G/B
for i, (label, color) in enumerate(zip(['DC Component R (Base Color)',
                                          'DC Component G (Base Color)',
                                          'DC Component B (Base Color)'],
                                         ['red','green','blue'])):
    ax = fig5.add_subplot(gs[0, i])
    ax.hist(f_dc[:, i], bins=80, color=color, alpha=0.7)
    ax.axvline(f_dc[:, i].mean(), color='k', linestyle='--',
               label=f'mean={f_dc[:,i].mean():.4f}')
    ax.set_title(f"{label}\nmin={f_dc[:,i].min():.4f} max={f_dc[:,i].max():.4f}")
    ax.set_xlabel('value'); ax.set_ylabel('count')
    ax.legend(fontsize=8)

# SH degree별 에너지
all_sh  = np.concatenate([f_dc, f_rest], axis=1)               # (N, 48)
sh_3ch  = all_sh.reshape(N, 16, 3)
deg_energy = []
deg_sizes  = [1, 3, 5, 7]  # degree 0,1,2,3의 계수 수
idx = 0
for deg, size in enumerate(deg_sizes):
    e = np.abs(sh_3ch[:, idx:idx+size, :]).mean()
    deg_energy.append(e)
    idx += size
deg_energy = np.array(deg_energy) / (deg_energy[0] + 1e-8)

ax_energy = fig5.add_subplot(gs[1, 0])
ax_energy.bar(['Deg 0 (DC)', 'Deg 1', 'Deg 2', 'Deg 3'],
              deg_energy, color=['gold','orange','darkorange','red'])
ax_energy.set_title("Mean Energy per SH Degree")
ax_energy.set_xlabel('SH Degree'); ax_energy.set_ylabel('mean |coeff|')

# 전체 SH 계수 분포
ax_full = fig5.add_subplot(gs[1, 1])
ax_full.hist(all_sh.ravel(), bins=100, color='purple', alpha=0.7)
ax_full.set_title(f"Full SH Coefficient Distribution\nshape={all_sh.shape}")
ax_full.set_xlabel('coefficient value'); ax_full.set_ylabel('count')

# 3D 포인트 클라우드 (DC 색상)
ax_3d2 = fig5.add_subplot(gs[1, 2], projection='3d')
ax_3d2.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
               c=colors, s=0.3, alpha=0.5)
ax_3d2.set_title("3D Point Cloud (DC Color)")
ax_3d2.set_xlabel('X'); ax_3d2.set_ylabel('Y'); ax_3d2.set_zlabel('Z')

plt.tight_layout()
plt.savefig(save_dir / f"{save_name}_5_features_sh.png", dpi=120, bbox_inches='tight')
plt.close()
print("저장: vis_5_features_sh.png")

print(f"\n전체 완료! 저장 위치: {save_dir}")