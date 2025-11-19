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

THRESHOLD  = 0.3                                # floor_prob 임계

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

def compute_plane_corners_pca(xyz_inliers, model, expand_ratio=1.1):
    """PCA 기반 평면 사각형 꼭짓점 계산"""
    n = model[:3] / np.linalg.norm(model[:3])
    d = float(model[3])
    center = xyz_inliers.mean(axis=0)

    # 평면에 투영
    xyz_proj = xyz_inliers - (xyz_inliers @ n + d)[:, None] * n

    # SVD로 평면상 주축(u,v) 계산
    X = xyz_proj - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]; v = Vt[1]
    u /= np.linalg.norm(u)
    v -= (v @ u) * u
    v /= np.linalg.norm(v)

    # 평면상 좌표로 변환
    uv = np.stack([X @ u, X @ v], axis=1)
    min_u, min_v = uv.min(axis=0)
    max_u, max_v = uv.max(axis=0)

    du = (max_u - min_u) * expand_ratio
    dv = (max_v - min_v) * expand_ratio
    cu = (max_u + min_u) * 0.5
    cv = (max_v + min_v) * 0.5

    min_u, max_u = cu - du * 0.5, cu + du * 0.5
    min_v, max_v = cv - dv * 0.5, cv + dv * 0.5

    # 사각형 생성
    corners_local = np.array([
        [min_u, min_v],
        [max_u, min_v],
        [max_u, max_v],
        [min_u, max_v],
    ], dtype=np.float64)
    corners_world = center + corners_local[:, 0:1] * u + corners_local[:, 1:1+1] * v

    # CCW 정렬
    rel = corners_world - corners_world.mean(axis=0)
    angles = np.arctan2(rel @ v, rel @ u)
    order = np.argsort(angles)
    corners_world = corners_world[order]

    return corners_world

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

    # PLY 로드 → floor 서브셋 만들기
    floor_xyz, floor_vtable, floor_ply = read_xyz_and_vertex(FLOOR_PATH)
    N = len(floor_xyz)
    print(f"[LOAD] floor.ply: {N} points")

    # RANSAC
    inliers, model = ransac_floor(floor_xyz)

    # 평면에 속하는 스플랫 점 (inlier) 추출
    floor_points = floor_xyz[inliers]

    corners = compute_plane_corners_pca(floor_xyz[inliers], model)

    print("Corners : ", corners)

    ### 전체 모델
    # CSV에서 floor 인덱스 뽑기
    arr = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    if arr.ndim == 1: 
        arr = arr[None,:]
    idx = arr[:,0].astype(int)
    prob = arr[:,1].astype(float)
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