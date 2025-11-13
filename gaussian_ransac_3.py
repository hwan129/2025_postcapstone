# input : 바닥만 있는 .ply, output : 추정된 평면의 촤표, 바닥이 평평한 전체 가우시안 모델
import numpy as np
from pathlib import Path
import open3d as o3d
from plyfile import PlyData, PlyElement
import argparse, os, sys
from pathlib import Path

import time

start = time.perf_counter()

# python gaussan_ransac_3.py -d output/oseok4(gaussian model)
parser = argparse.ArgumentParser(description="평면 추정")
parser.add_argument("-d", "--data", required=True, help="Gaussian Model")
args = parser.parse_args()

CSV_PATH   = os.path.join(args.data, "floor_data/gaussian_floor_prob.csv")
FLOOR_PATH   = os.path.join(args.data, "floor_data/floor.ply")
GAUSSIAN_PATH   = os.path.join(args.data, "point_cloud/iteration_30000/point_cloud.ply")
OUT_DIR = Path(args.data) / "floor_data"

THRESHOLD  = 0.3                                   # floor_prob 임계

# RANSAC 파라미터 (네 코드와 동일/유사)
DIST_TH     = 0.03
RANSAC_N    = 3
N_ITER      = 2000
MIN_INLIER_RATIO = 0.002
UP_AXIS     = np.array([0, 1, 0], float)  # z-up이면 [0,0,1], 월드 업 축
ANGLE_TOL   = 5.0 # 이 각도 이하는 평면으로 인정

# 벡터 정규화?
def _unit(v):
    n = np.linalg.norm(v); return v/n if n>0 else v

# 법선 n과 기준축 axis 사이의 끼인 각 계산
def _angle_deg_to_axis(n, axis):
    n = _unit(np.asarray(n,float)); axis = _unit(np.asarray(axis,float))
    c = float(np.clip(abs(np.dot(n, axis)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

# ply 읽기
def read_xyz_and_vertex(p):
    ply = PlyData.read(p)
    v = ply['vertex']
    xyz = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
    return xyz, v.data, ply

# 원본 ply 형식을 유지하여 ply 파일 만들기
def write_subset(path, ply_full, v_subset):
    out = PlyData([PlyElement.describe(v_subset, 'vertex')], text=ply_full.text, byte_order=ply_full.byte_order)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(path.as_posix())

# 평면 추정
def ransac_floor(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    original = pcd; current = pcd; cur_idx = np.arange(len(xyz))
    planes=[]
    while len(current.points) >= RANSAC_N:
        model, inliers = current.segment_plane(DIST_TH, RANSAC_N, N_ITER)
        inliers = np.asarray(inliers, np.int64)
        if inliers.size < max(1, int(len(current.points)*MIN_INLIER_RATIO)): break
        orig_inl = cur_idx[inliers]
        n = np.asarray(model[:3], float)
        ang = _angle_deg_to_axis(n, UP_AXIS)
        planes.append((model, orig_inl, ang))
        remain = np.setdiff1d(cur_idx, orig_inl)
        current = original.select_by_index(remain); cur_idx = remain
    if not planes: return None, None
    floor_cands = [p for p in planes if p[2] <= ANGLE_TOL]
    best = max(floor_cands or planes, key=lambda x: len(x[1]))
    return np.unique(best[1]), best[0]

# 평면 꼭짓점 계산
def compute_plane_corners(points, normal):
    """
    points: 평면 위 점들 (Nx3)
    normal: 단위 법선 벡터 (3,)
    return: corners (4x3)
    """
    # 1. 법선에 수직인 두 축(u, v) 계산
    normal = normal / np.linalg.norm(normal)
    # u축: 법선과 다른 임의 벡터의 외적
    ref = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    # 2. 점들을 (u,v) 평면 좌표계로 변환
    proj_u = np.dot(points, u)
    proj_v = np.dot(points, v)

    # 3. 2D bounding box 계산
    umin, umax = proj_u.min(), proj_u.max()
    vmin, vmax = proj_v.min(), proj_v.max()

    # 4. 꼭짓점을 3D로 복원
    corners = [
        u * umin + v * vmin,
        u * umax + v * vmin,
        u * umax + v * vmax,
        u * umin + v * vmax
    ]
    # 중심점(원점 기준) 보정
    center = points.mean(axis=0)
    corners = [center + c for c in corners]

    return np.array(corners)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # PLY 로드 → floor 서브셋 만들기
    floor_xyz, floor_vtable, floor_ply = read_xyz_and_vertex(FLOOR_PATH)
    N = len(floor_xyz)
    print(f"[LOAD] floor.ply: {N} points")

    # RANSAC
    inliers, model = ransac_floor(floor_xyz)

    # 평면 정보
    a, b, c, d = model
    n = np.array([a, b, c], dtype=float)
    n_norm = np.linalg.norm(n)
    n_unit = n / n_norm # 법선
    distance = -d / n_norm  # 원점과 평면의 거리

    print("model (a,b,c,d):", model)
    print("normal (x, y, z):", n_unit.tolist())
    print("distance :", float(distance))

    # 평면에 속하는 스플랫 점 (inlier) 추출
    floor_points = floor_xyz[inliers]

    # x,z축 기준으로 최대 최소점 계산
    x_min, y_min, z_min = floor_points.min(axis=0)
    x_max, y_max, z_max = floor_points.max(axis=0)

    print(f"[BOUNDING BOX] x_min={x_min:.3f}, x_max={x_max:.3f}, "
          f"z_min={z_min:.3f}, z_max={z_max:.3f}")

    # 꼭짓점 계산 (x,z축 평면 상 사각형)
    corners = np.array([
        [x_min, (y_min + y_max) / 2, z_min],
        [x_max, (y_min + y_max) / 2, z_min],
        [x_max, (y_min + y_max) / 2, z_max],
        [x_min, (y_min + y_max) / 2, z_max],
    ])
    print("[CORNERS]")
    for i, c in enumerate(corners):
        print(f"corner_{i+1}: {c.tolist()}")

    ### 전체 모델
    # CSV에서 floor 인덱스 뽑기
    arr = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    if arr.ndim == 1: arr = arr[None,:]
    idx = arr[:,0].astype(int); prob = arr[:,1].astype(float)
    sel = idx[prob >= THRESHOLD]

    # PLY 로드 → floor 서브셋 만들기
    all_xyz, all_vtable, all_ply = read_xyz_and_vertex(GAUSSIAN_PATH)

    if np.dot(n, UP_AXIS) < 0:
        n = -n
        d = -d

    fulls_floor_xyz = all_xyz[sel]
    splats = fulls_floor_xyz @ n + d
    above_mask = splats < 0

    xyz_new = all_xyz.copy()
    n_unit = n / np.linalg.norm(n)

    for i, idx in enumerate(sel[np.where(above_mask)[0]]):
        dist = np.dot(n_unit, all_xyz[idx]) + d
        xyz_new[idx] = all_xyz[idx] - n_unit * dist

    # 새로운 좌표
    v_new = all_vtable.copy()
    v_new['x'] = xyz_new[:, 0]
    v_new['y'] = xyz_new[:, 1]
    v_new['z'] = xyz_new[:, 2]

    out_ply = OUT_DIR / "projection.ply"
    write_subset(out_ply, all_ply, v_new)

    print(f"[DONE] saved projected scene to {out_ply}")

if __name__ == "__main__":
    main()

end = time.perf_counter()

print(f"실행 시간: {end - start:.6f}초")