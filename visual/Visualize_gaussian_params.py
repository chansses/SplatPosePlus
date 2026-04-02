import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load parameters
params = np.load("results/01Gorilla/gaussian_params.npy", allow_pickle=True).item()
xyz      = params["xyz"]        # (N, 3)
features = params["features"]   # (N, 16, 3)
scaling  = params["scaling"]    # (N, 3)
rotation = params["rotation"]   # (N, 4)
opacity  = params["opacity"]    # (N, 1)

N = xyz.shape[0]
print(f"Total Gaussians: {N}")

# ── Figure 1: XYZ 3D Position ─────────────────────────────
fig = plt.figure(figsize=(18, 5))
fig.suptitle(f"[1] XYZ - 3D Position  (N={N})", fontsize=14, fontweight='bold')

ax1 = fig.add_subplot(141, projection='3d')
sc = ax1.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
                 c=opacity[:,0], cmap='plasma', s=0.5, alpha=0.5)
ax1.set_title("3D Point Cloud\n(color=opacity)")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
plt.colorbar(sc, ax=ax1, shrink=0.5)

for i, (label, color) in enumerate(zip(['X','Y','Z'], ['r','g','b'])):
    ax = fig.add_subplot(1, 4, i+2)
    ax.hist(xyz[:,i], bins=80, color=color, alpha=0.7, edgecolor='none')
    ax.axvline(xyz[:,i].mean(), color='black', linestyle='--',
               label=f'mean={xyz[:,i].mean():.3f}')
    ax.set_title(f"{label}-axis Distribution\nmin={xyz[:,i].min():.3f}  max={xyz[:,i].max():.3f}")
    ax.set_xlabel(label); ax.set_ylabel("count"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("vis_1_xyz.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: vis_1_xyz.png")

# ── Figure 2: Scaling ─────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("[2] Scaling - Gaussian Size", fontsize=14, fontweight='bold')

for i, (label, color) in enumerate(zip(['X','Y','Z'], ['r','g','b'])):
    axes[i].hist(scaling[:,i], bins=80, color=color, alpha=0.7, edgecolor='none')
    axes[i].axvline(scaling[:,i].mean(), color='black', linestyle='--',
                    label=f'mean={scaling[:,i].mean():.4f}')
    axes[i].set_title(f"Scale {label}\nmin={scaling[:,i].min():.4f}  max={scaling[:,i].max():.4f}")
    axes[i].set_xlabel("scale"); axes[i].set_ylabel("count"); axes[i].legend(fontsize=8)

avg_scale = scaling.mean(axis=1)
axes[3].hist(avg_scale, bins=80, color='purple', alpha=0.7, edgecolor='none')
axes[3].axvline(avg_scale.mean(), color='black', linestyle='--',
                label=f'mean={avg_scale.mean():.4f}')
axes[3].set_title(f"Scale Mean (3-axis)\nmin={avg_scale.min():.4f}  max={avg_scale.max():.4f}")
axes[3].set_xlabel("avg scale"); axes[3].set_ylabel("count"); axes[3].legend(fontsize=8)

plt.tight_layout()
plt.savefig("vis_2_scaling.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: vis_2_scaling.png")

# ── Figure 3: Opacity ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("[3] Opacity", fontsize=14, fontweight='bold')

op = opacity[:,0]
axes[0].hist(op, bins=80, color='steelblue', alpha=0.7, edgecolor='none')
axes[0].axvline(op.mean(), color='black', linestyle='--', label=f'mean={op.mean():.4f}')
axes[0].set_title(f"Opacity Distribution\nmin={op.min():.4f}  max={op.max():.4f}")
axes[0].set_xlabel("opacity"); axes[0].set_ylabel("count"); axes[0].legend(fontsize=8)

axes[1].hist(op, bins=80, color='steelblue', alpha=0.7,
             cumulative=True, density=True, edgecolor='none')
axes[1].set_title("Opacity Cumulative Distribution (CDF)")
axes[1].set_xlabel("opacity"); axes[1].set_ylabel("cumulative ratio")

thresholds = np.linspace(0, 1, 50)
counts = [(op > t).sum() for t in thresholds]
axes[2].plot(thresholds, counts, color='steelblue', linewidth=2)
axes[2].set_title("Active Gaussians per Threshold")
axes[2].set_xlabel("opacity threshold"); axes[2].set_ylabel("count"); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("vis_3_opacity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: vis_3_opacity.png")

# ── Figure 4: Rotation (Quaternion) ──────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("[4] Rotation - Quaternion (qw, qx, qy, qz)", fontsize=14, fontweight='bold')

for i, (label, color) in enumerate(zip(['qw','qx','qy','qz'], ['gold','r','g','b'])):
    axes[i].hist(rotation[:,i], bins=80, color=color, alpha=0.7, edgecolor='none')
    axes[i].axvline(rotation[:,i].mean(), color='black', linestyle='--',
                    label=f'mean={rotation[:,i].mean():.4f}')
    axes[i].set_title(f"{label}\nstd={rotation[:,i].std():.4f}")
    axes[i].set_xlabel(label); axes[i].set_ylabel("count"); axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig("vis_4_rotation.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: vis_4_rotation.png")

# ── Figure 5: Spherical Harmonics ────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.suptitle("[5] Spherical Harmonics (features) - Color Representation", fontsize=14, fontweight='bold')

# DC component (0th order SH) → base color
dc = features[:, 0, :]  # (N, 3)
for i, (label, color) in enumerate(zip(['R','G','B'], ['r','g','b'])):
    ax = fig.add_subplot(2, 3, i+1)
    ax.hist(dc[:,i], bins=80, color=color, alpha=0.7, edgecolor='none')
    ax.axvline(dc[:,i].mean(), color='black', linestyle='--',
               label=f'mean={dc[:,i].mean():.4f}')
    ax.set_title(f"DC Component {label} (Base Color)\nmin={dc[:,i].min():.4f}  max={dc[:,i].max():.4f}")
    ax.set_xlabel("value"); ax.set_ylabel("count"); ax.legend(fontsize=8)

# SH energy per degree
sh_labels = ['Deg 0 (DC)', 'Deg 1', 'Deg 2', 'Deg 3']
sh_energy, start = [], 0
for n in [1, 3, 5, 7]:
    sh_energy.append(np.abs(features[:, start:start+n, :]).mean())
    start += n

ax = fig.add_subplot(2, 3, 4)
ax.bar(sh_labels, sh_energy, color=['gold','orange','tomato','crimson'], alpha=0.8)
ax.set_title("Mean Energy per SH Degree")
ax.set_xlabel("SH Degree"); ax.set_ylabel("mean |coeff|")

# Full coefficient distribution
ax = fig.add_subplot(2, 3, 5)
ax.hist(features.ravel(), bins=100, color='purple', alpha=0.7, edgecolor='none')
ax.set_title(f"Full SH Coefficient Distribution\nshape={features.shape}")
ax.set_xlabel("coefficient value"); ax.set_ylabel("count")

# 3D scatter with DC color
dc_norm = (dc - dc.min(0)) / (dc.max(0) - dc.min(0) + 1e-8)
ax3d = fig.add_subplot(2, 3, 6, projection='3d')
ax3d.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=dc_norm, s=0.5, alpha=0.6)
ax3d.set_title("3D Point Cloud (DC Color)")
ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

plt.tight_layout()
plt.savefig("vis_5_features_sh.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: vis_5_features_sh.png")

# ── Summary Statistics ────────────────────────────────────
print("\n" + "="*55)
print("Parameter Summary Statistics")
print("="*55)
for name, val in params.items():
    flat = val.ravel()
    print(f"\n[{name}]  shape={val.shape}")
    print(f"  mean   = {flat.mean():.6f}")
    print(f"  std    = {flat.std():.6f}")
    print(f"  min    = {flat.min():.6f}")
    print(f"  max    = {flat.max():.6f}")
    print(f"  median = {np.median(flat):.6f}")

print("\nAll visualizations saved!")
print("  vis_1_xyz.png")
print("  vis_2_scaling.png")
print("  vis_3_opacity.png")
print("  vis_4_rotation.png")
print("  vis_5_features_sh.png")