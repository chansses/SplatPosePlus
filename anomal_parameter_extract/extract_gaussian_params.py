import torch
import numpy as np
from gaussian_splatting.gaussian_renderer import GaussianModel

# 클래스 및 경로 설정
classname = "01Gorilla"
iters = 15000
ply_path = f"results/{classname}/point_cloud/iteration_{iters}/point_cloud.ply"  # 정상(normal) 객체의 3D 외관을 Gaussian으로 표현

# 모델 로드
gs = GaussianModel(3)
gs.load_ply(ply_path)

# 파라미터 추출
params = {
    "xyz"      : gs.get_xyz.detach().cpu().numpy(),       # (N, 3) 3D 위치
    "features" : gs.get_features.detach().cpu().numpy(),  # (N, 16, 3) 색상 SH
    "scaling"  : gs.get_scaling.detach().cpu().numpy(),   # (N, 3) 크기
    "rotation" : gs.get_rotation.detach().cpu().numpy(),  # (N, 4) 회전 quaternion
    "opacity"  : gs.get_opacity.detach().cpu().numpy(),   # (N, 1) 불투명도
}

# 확인 출력
print(f"총 Gaussian 수: {params['xyz'].shape[0]}")
for k, v in params.items():
    print(f"  {k:10s}: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}")

# 저장
np.save(f"results/{classname}/gaussian_params.npy", params)
print(f"\n저장 완료: results/{classname}/gaussian_params.npy")

params = np.load("results/01Gorilla/gaussian_params.npy", allow_pickle=True).item()

xyz      = params["xyz"]       # (N, 3)
features = params["features"]  # (N, 16, 3)
scaling  = params["scaling"]   # (N, 3)
rotation = params["rotation"]  # (N, 4)
opacity  = params["opacity"]   # (N, 1)