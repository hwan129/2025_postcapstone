# 추정된 바닥보다 위에 있는 스플랫을 평면으로 투영 후 전체 모델이 output

import numpy as np
from pathlib import Path
import open3d as o3d
from plyfile import PlyData, PlyElement

import time

start = time.perf_counter()

CSV_PATH   = Path("../SegAnyGAussians/output/oseok4/point_cloud/iteration_30000/gaussian_floor_prob.csv")   # 네가 만든 CSV
PLY_PATH   = Path("../SegAnyGAussians/output/oseok4/point_cloud/iteration_30000/scene_point_cloud.ply")     # 원본 가우시안 PLY
OUT_DIR    = Path("output/gaussian")
THRESHOLD  = 0.5                                   # floor_prob 임계

# RANSAC 파라미터 (네 코드와 동일/유사)
DIST_TH     = 0.03
RANSAC_N    = 3
N_ITER      = 2000
MIN_INLIER_RATIO = 0.002
UP_AXIS     = np.array([0, 1, 0], float)  # z-up이면 [0,0,1]
ANGLE_TOL   = 5.0

def _unit(v):
    n = np.linalg.norm(v); return v/n if n>0 else v

def _angle_deg_to_axis(n, axis):
    n = _unit(np.asarray(n,float)); 
    axis = _unit(np.asarray(axis,float))
    c = float(np.clip(abs(np.dot(n, axis)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def read_xyz_and_vertex(p):
    ply = PlyData.read(p)
    v = ply['vertex']
    xyz = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
    return xyz, v.data, ply

def write_subset(path, ply_full, v_subset):
    out = PlyData([PlyElement.describe(v_subset, 'vertex')], text=ply_full.text, byte_order=ply_full.byte_order)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(path.as_posix())

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

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 1) CSV에서 floor 인덱스 뽑기
    arr = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    if arr.ndim == 1: arr = arr[None,:]
    idx = arr[:,0].astype(int)
    prob = arr[:,1].astype(float)
    sel = idx[prob >= THRESHOLD]

    # 2) PLY 로드 → floor 서브셋 만들기
    xyz, vtable, ply_full = read_xyz_and_vertex(PLY_PATH)
    if sel.size == 0:
        print("[WARN] no floor candidates at this threshold")
        return

    # v_floor = vtable[sel]
    # floor_only_path = OUT_DIR / f"floor_candidates_thr{THRESHOLD:.2f}.ply"
    # write_subset(floor_only_path, ply_full, v_floor)
    # print("[SAVE] candidates:", floor_only_path, "(N=", len(v_floor), ")")

    # 3) RANSAC (floor-only 후보에 대해서 실행 → 훨씬 빨라짐)
    xyz_floor = xyz[sel]
    inliers, model = ransac_floor(xyz_floor)
    if inliers is None or inliers.size == 0:
        print("[WARN] RANSAC found no plane")
        return

    n = np.asarray(model[:3], float)
    d = float(model[3])

    if np.dot(n, UP_AXIS) < 0:
        n = -n
        d = -d

    # 4 평면 보다 위아래 판별
    splats = xyz_floor @ n + d
    above_mask = splats < 0

    xyz_new = xyz.copy()
    n_unit = n / np.linalg.norm(n)

    for i, idx in enumerate(sel[np.where(above_mask)[0]]):
        dist = np.dot(n_unit, xyz[idx]) + d
        xyz_new[idx] = xyz[idx] - n_unit * dist

    # 5 새로운 좌표
    v_new = vtable.copy()
    v_new['x'] = xyz_new[:, 0]
    v_new['y'] = xyz_new[:, 1]
    v_new['z'] = xyz_new[:, 2]

    out_ply = OUT_DIR / "projection.ply"
    write_subset(out_ply, ply_full, v_new)

    print(f"[DONE] saved projected scene to {out_ply}")

    # 4) 최종 평면 인라이어만 저장
    # final_idx_in_full = sel[inliers]  # 원본 인덱스 공간으로 변환
    # final_subset = vtable[final_idx_in_full]
    # out_ply = OUT_DIR / "floor_plane_inliers.ply"
    # write_subset(out_ply, ply_full, final_subset)
    # print("[DONE] floor inliers:", len(final_subset), "->", out_ply)
    # print("[PLANE] model (a,b,c,d):", model)

if __name__ == "__main__":
    main()

end = time.perf_counter()

print(f"실행 시간: {end - start:.6f}초")