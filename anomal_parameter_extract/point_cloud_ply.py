from plyfile import PlyData
import numpy as np
import matplotlib.pyplot as plt

ply = PlyData.read("results/11Pig/point_cloud/iteration_15000/point_cloud.ply")
v = ply['vertex']

x, y, z = v['x'], v['y'], v['z']

opacity = 1 / (1 + np.exp(-v['opacity']))
mask = opacity > 0.1
x, y, z = x[mask], y[mask], z[mask]

r = 0.5 + 0.282 * v['f_dc_0'][mask]
g = 0.5 + 0.282 * v['f_dc_1'][mask]
b = 0.5 + 0.282 * v['f_dc_2'][mask]
colors = np.clip(np.stack([r, g, b], axis=1), 0, 1)

print(f"총 점 수: {len(x)}개 (opacity > 0.1)")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('black')

views = [
    (x, z, 'Front (X-Z)'),
    (x, y, 'Top (X-Y)'),
    (y, z, 'Side (Y-Z)'),
]

for ax, (a, b_ax, title) in zip(axes, views):
    ax.set_facecolor('black')
    ax.scatter(a, b_ax, c=colors, s=0.3, alpha=0.6)
    ax.set_title(title, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

plt.suptitle('11Pig - 3DGS Point Cloud', color='white', fontsize=14)
plt.tight_layout()
plt.savefig('results/11Pig/point_cloud_vis.png', dpi=150, bbox_inches='tight', facecolor='black')
print("저장 완료: results/11Pig/point_cloud_vis.png")