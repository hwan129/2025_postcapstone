# saga에서 segementation 결과 .pt를 .ply로 변환

# input -> .pt(선택된 스플랫인지 아닌지, saga의 output임), 원본.ply
# output -> projection된 gaussian splatting.ply, 추정된 평면의 네개 꼭짓점, 비바닥.ply,

import torch
import numpy as np
import open3d as o3d   
UP_AXIS = np.array([0,1,0], float)

# INPUT_PLY = "output/oseok3/point_cloud/iteration_30000/scene_point_cloud.ply"
INPUT_PLY = "./segment/scene_point_cloud.ply"
MASK_PATH = "./segmentation_res/oseok3.pt"
OUTPUT_PLY = "./segment/oseok3.ply"

# saga에서 나온 .pt 파일 불러오기
def load_ply_binary(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii")
            header.append(line)
            if line.strip() == "end_header":
                break
        
        # vertex count 찾기
        for h in header:
            if h.startswith("element vertex"):
                vertex_count = int(h.split()[-1])
        
        # 한 vertex의 property 개수 = float property 개수
        # header에서 property float count 하기
        prop_lines = [h for h in header if h.startswith("property float")]
        prop_count = len(prop_lines)

        # binary 데이터 읽기
        # float32 개수 = vertex_count * prop_count
        data = np.fromfile(f, dtype=np.float32, count=vertex_count * prop_count)
        data = data.reshape((vertex_count, prop_count))

    return header, data, vertex_count, prop_count

# ply 파일 저장하기
def save_ply_binary(path, header, data):
    # header 수정: element vertex N 값 바꾸기
    new_header = []
    for h in header:
        if h.startswith("element vertex"):
            new_header.append(f"element vertex {len(data)}\n")
        else:
            new_header.append(h)

    with open(path, "wb") as f:
        # header 저장
        for h in new_header:
            f.write(h.encode("ascii"))
        
        # binary 데이터 저장
        data.astype(np.float32).tofile(f)

# Load original PLY
header, data, vertex_count, prop_count = load_ply_binary(INPUT_PLY) # saga에서 나온 .pt 파일 불러오기
print(f"Loaded {vertex_count} vertices with {prop_count} properties per vertex.")

# Load segmentation mask
# mask = torch.load(MASK_PATH).view(-1).cpu().numpy()
# assert len(mask) == vertex_count, "mask length != vertex count"

# # Ture : 선택된 스플랫만(바닥만), False : 선택 안된 애들(비바닥)
# nonfloor_data = data[mask == False] 
# floor_data = data[mask == True] 
# Load segmentation mask
mask = torch.load(MASK_PATH).view(-1).cpu().numpy()

# 1) 반전 여부 체크 (비율 이상하면 자동 뒤집기)
true_count = np.sum(mask == True)
false_count = np.sum(mask == False)

if true_count < false_count * 0.1:   # floor가 전체의 10% 미만이면 반전
    print("WARNING: mask too small → flipping mask.")
    mask = ~mask

# 2) dilation으로 floor 확장
from scipy.ndimage import binary_dilation
mask = binary_dilation(mask, iterations=3).astype(bool)

# 이제 floor_data, nonfloor_data 다시 계산
nonfloor_data = data[mask == False]
floor_data = data[mask == True]


# 비바닥.ply 저장
print(f"Selected {len(nonfloor_data)} / {vertex_count} vertices")
save_ply_binary(OUTPUT_PLY, header, nonfloor_data)
print(f"Saved filtered PLY → {OUTPUT_PLY}")

# ransac
import open3d as o3d

PROJECTION_PLY = "./output/oseok3_projection.ply"
PLANE_CORNERS_TXT = "./output/oseok3_plane_corners.txt"

# 란삭을 위한 파라미터
DIST_TH     = 0.03
RANSAC_N    = 3
N_ITER      = 2000
MIN_INLIER_RATIO = 0.002
UP_AXIS     = np.array([0, 1, 0], float)  # z-up이면 [0,0,1]로 수정
ANGLE_TOL   = 5.0  # 이 각도 이하는 바닥 평면

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _angle_deg_to_axis(n, axis):
    n = _unit(np.asarray(n, float))
    axis = _unit(np.asarray(axis, float))
    c = float(np.clip(abs(np.dot(n, axis)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def ransac_floor(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    original = pcd
    current = pcd
    cur_idx = np.arange(len(xyz))
    planes = []

    while len(current.points) >= RANSAC_N:
        model, inliers = current.segment_plane(
            distance_threshold=DIST_TH,
            ransac_n=RANSAC_N,
            num_iterations=N_ITER
        )
        inliers = np.asarray(inliers, np.int64)
        if inliers.size < max(1, int(len(current.points) * MIN_INLIER_RATIO)):
            break

        orig_inl = cur_idx[inliers]
        n = np.asarray(model[:3], float)
        ang = _angle_deg_to_axis(n, UP_AXIS)
        planes.append((model, orig_inl, ang))

        remain = np.setdiff1d(cur_idx, orig_inl)
        current = original.select_by_index(remain)
        cur_idx = remain

    if not planes:
        return None, None

    floor_cands = [p for p in planes if p[2] <= ANGLE_TOL]
    best = max(floor_cands or planes, key=lambda x: len(x[1]))
    return np.unique(best[1]), best[0]

def compute_plane_corners_pca(xyz_inliers, model, expand_ratio=1.1):
    n = model[:3] / np.linalg.norm(model[:3])
    d = float(model[3])
    center = xyz_inliers.mean(axis=0)

    # 평면에 투영
    xyz_proj = xyz_inliers - (xyz_inliers @ n + d)[:, None] * n

    # SVD로 평면상 주축(u, v) 계산
    X = xyz_proj - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]
    v = Vt[1]
    u /= np.linalg.norm(u)
    v -= (v @ u) * u
    v /= np.linalg.norm(v)

    # 평면상 좌표계로 변환
    uv = np.stack([X @ u, X @ v], axis=1)
    min_u, min_v = uv.min(axis=0)
    max_u, max_v = uv.max(axis=0)

    du = (max_u - min_u) * expand_ratio
    dv = (max_v - min_v) * expand_ratio
    cu = (max_u + min_u) * 0.5
    cv = (max_v + min_v) * 0.5

    min_u, max_u = cu - du * 0.5, cu + du * 0.5
    min_v, max_v = cv - dv * 0.5, cv + dv * 0.5

    corners_local = np.array([
        [min_u, min_v],
        [max_u, min_v],
        [max_u, max_v],
        [min_u, max_v],
    ], dtype=np.float64)

    corners_world = center + corners_local[:, 0:1] * u + corners_local[:, 1:2] * v

    # CCW 정렬
    rel = corners_world - corners_world.mean(axis=0)
    angles = np.arctan2(rel @ v, rel @ u)
    order = np.argsort(angles)
    corners_world = corners_world[order]

    return corners_world

# 1) 바닥 포인트만 사용해서 평면 RANSAC
floor_xyz = floor_data[:, :3].astype(np.float64)
all_xyz = data[:, :3].astype(np.float64)

inliers, model = ransac_floor(floor_xyz)
if inliers is None:
    raise RuntimeError("바닥 평면을 찾지 못했습니다.")

a, b, c, d = model
n = np.array([a, b, c], dtype=float)

# 월드 up 축과 같은 방향으로 노멀 정렬
if np.dot(n, UP_AXIS) < 0:
    n = -n
    d = -d

n_unit = n / np.linalg.norm(n)

print(f"Estimated plane: {n[0]:.6f} x + {n[1]:.6f} y + {n[2]:.6f} z + {d:.6f} = 0")

# 2) 평면 위의 인라이어로부터 네 개 꼭짓점 추정
floor_inlier_points = floor_xyz[inliers]
corners = compute_plane_corners_pca(
    floor_inlier_points,
    np.array([n[0], n[1], n[2], d], dtype=float)
)
print("Floor plane corners (x, y, z):")
print(corners)

# 텍스트 파일로도 저장
np.savetxt(PLANE_CORNERS_TXT, corners, fmt="%.6f", delimiter=",")
print(f"Saved plane corners → {PLANE_CORNERS_TXT}")

# 3) 전체 가우시안에서 '바닥으로 분류된 포인트'만 평면으로 정사영
floor_idx = np.where(mask == True)[0]

# 전체 좌표 복사
xyz_new = all_xyz.copy()

# 평면: n·x + d = 0
n_norm = np.linalg.norm(n)
n_unit = n / n_norm

# 각 바닥 포인트의 signed distance (거리 방향은 n_unit)
# dist = (n·p + d) / ||n||
floor_pts = all_xyz[floor_idx]
signed_dists = (floor_pts @ n + d) / n_norm  # shape: (N_floor,)

# projection: p_proj = p - dist * n_unit
signed_dists = (floor_pts @ n + d) / n_norm

# 평면 위(+) 에 있는 애들만 projection
above_mask = signed_dists > 0

proj_idx = floor_idx[above_mask]

xyz_new[proj_idx] = all_xyz[proj_idx] - signed_dists[above_mask][:, None] * n_unit[None, :]


# 4) 정사영된 좌표로 PLY 저장
projected_data = data.copy()
projected_data[:, 0] = xyz_new[:, 0]
projected_data[:, 1] = xyz_new[:, 1]
projected_data[:, 2] = xyz_new[:, 2]

save_ply_binary(PROJECTION_PLY, header, projected_data)
print(f"Saved projected gaussian splatting PLY → {PROJECTION_PLY}")